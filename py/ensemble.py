import os
import joblib
import numpy as np
import warnings
from joblib import Parallel
from scipy.sparse import issparse

from sklearn.utils import check_random_state
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn.utils.multiclass import type_of_target
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble._forest import BaseForest, MAX_INT, _parallel_build_trees, _get_n_samples_bootstrap
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.preprocessing import StandardScaler


class ProbabilisticRandomForestRegressor(RandomForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=-1 if n_jobs is None else n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

        self.output_transformer = None
        
    def save_model(self, filename):
        check_is_fitted(self)
        if os.path.exists(filename):
            raise FileExistsError
        joblib.dump(self, filename)
    
    @classmethod
    def load_model(self, filename):
        with warnings.catch_warnings():  # suppress ignoring scipy compatability warning
            warnings.simplefilter('ignore')
            self = joblib.load(filename)
            check_is_fitted(self) 
        return self
    
    def label_atleast2d(self, y):
        y = np.atleast_1d(y)
        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))
        return y

    def fit(self, X, y, Xerr=0., yerr=0., sample_weight=None):
        """
        See scikit-learn for docs, this function is from scikit-learn and prfr modification
        """
        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")
        X, y = self._validate_data(
            X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = self.label_atleast2d(y)
        
        if self.output_transformer is None:
            self.output_transformer = StandardScaler()
            self.output_transformer.fit(y)
            
        y = self.output_transformer.transform(y)
        yerr = yerr / self.output_transformer.scale_

        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        # Check parameters
        self._validate_estimator()
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warnings.warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                delayed(_parallel_build_trees)(
                    t,
                    self.bootstrap,
                    np.random.normal(X, Xerr),
                    y,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                )
                for i, t in enumerate(trees)
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)
            
        self.fit_bias(X, y, Xerr, yerr)

        if self.oob_score:
            y_type = type_of_target(y)
            if y_type in ("multiclass-multioutput", "unknown"):
                # FIXME: we could consider to support multiclass-multioutput if
                # we introduce or reuse a constructor parameter (e.g.
                # oob_score) allowing our user to pass a callable defining the
                # scoring strategy on OOB sample.
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )
            self._set_oob_score_and_attributes(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self
    
    def fit_bias(self, X, y, Xerr=0., yerr=0.):
        # y = self.output_transformer.transform(y)
        # yerr = yerr / self.output_transformer.scale_

        preds = self.predict(X, Xerr=Xerr)
        pred_mean = self.label_atleast2d(np.mean(preds, axis=-1))
        pred_mean = self.output_transformer.transform(pred_mean)
        pred_stdev = self.label_atleast2d(np.std(preds, axis=-1))
        pred_stdev /= self.output_transformer.scale_
        residuals = y - pred_mean
        self.bias_model = LinearRegression(fit_intercept=True, n_jobs=self.n_jobs)
        if np.any(yerr) == 0.:
            sample_weights=None
        else:
            sample_weights=np.sum(1./(yerr**2 + pred_stdev**2), axis=-1)
        self.bias_model.fit(X, residuals, sample_weight=sample_weights)
        
    def predict(self, X, Xerr=0.):
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Parallel loop
        y_hat = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(delayed(e.predict)(np.random.normal(X, Xerr)) for e in self.estimators_)
        y_hat = np.stack(y_hat).T
        if hasattr(self, "bias_model"):
            bias = self.bias_model.predict(X)
            y_hat += bias
        y_hat = self.output_transformer.inverse_transform(y_hat)
        return y_hat


class ProbabilisticRandomForestClassifier(RandomForestClassifier):
    def __init__(self):
        raise NotImplementedError
