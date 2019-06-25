#Assumes sklearn version 0.19.0

#Load Data
###############################################################################
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

#quickly making the target binary classification to simplify the example
y = 1*(y==0)

#Training and Testing Split
###############################################################################
from sklearn.model_selection import train_test_split
my_rand_state=0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                              random_state=my_rand_state)

#Define simple model
###############################################################################
from sklearn.linear_model import LogisticRegression
log_clf=LogisticRegression()
C=[0.001 , 0.01, 10, 100,1000]

#Simple pre-processing estimators
###############################################################################
from sklearn.preprocessing import StandardScaler
std_scale=StandardScaler()

#Defining the CV method: Using the Repeated Stratified K Fold
###############################################################################
from sklearn.model_selection import RepeatedStratifiedKFold
n_folds=10
n_repeats=30

skfold = RepeatedStratifiedKFold(n_splits=n_folds,n_repeats=n_repeats,
                                 random_state=my_rand_state)

#Creating simple pipeline and defining the gridsearch
###############################################################################
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
n_jobs=4

log_clf_pipe = Pipeline(steps=[('scale',std_scale),('clf',log_clf)])
log_clf_est = GridSearchCV(estimator=log_clf_pipe,cv=skfold,
              scoring='roc_auc',n_jobs=n_jobs,
              param_grid=dict(clf__C=C))

#Fit the Model & Plot the Results
###############################################################################
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

log_clf_est.fit(X_train,y_train)

#ploting results
log_fpr, log_tpr, _ = roc_curve(y_test,
                    log_clf_est.predict_proba(X_test)[:,1])
log_roc_auc = auc(log_fpr, log_tpr)

plt.plot(log_fpr, log_tpr, color='seagreen', linestyle='--',
         label='LOG (area = %0.2f)' % log_roc_auc, lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves of Model')
plt.legend(loc="lower right")
plt.show()