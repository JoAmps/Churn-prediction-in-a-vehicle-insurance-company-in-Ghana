
import logging
from data import process_data
from clean_data import load_data
from model import train_model, \
    compute_metrics, model_predictions, split_data
#from joblib import dump

logging.basicConfig(
    filename='log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

cat_features=['city','type_of_plan','highest_level_education','work_status',
    'sex','relationship_status','reachability','type_of_vehicle']



if __name__ == '__main__':
    df = load_data('cleaned_data.csv')
    train, test = split_data(df)
    X_train,y_train,lb,ohe,scaler=process_data(train,training=True,label='churn',cat_features=cat_features,)
    X_test,y_test,lb_t,ohe_t,scaler_t=process_data(test,training=False,label='churn',cat_features=cat_features,ohe=ohe, lb=lb, scaler=scaler)
    model = train_model(X_train, y_train)
    predictions = model_predictions(X_test, model)
    precision, recall, f1 = compute_metrics(y_test, predictions)
    model_scores =[] 
    scores="precision: %s " \
                "recall: %s f1: %s" %(precision, recall,f1 )
    model_scores.append(scores)            
    with open('model_metrics.txt', 'w') as out:
               for score in model_scores:
                out.write(score)
