# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.
from __future__ import print_function
import io
import json
import math
import os
import signal
import sys
import traceback
import flask
import pandas as pd
import joblib
import math
prefix = "/opt/ml/"
#prefix = "/Desktop/Walnut_OfferAcceptance" # path of jdirectory where joblib file is
model_path = os.path.join(prefix, "terms_ranking_model.joblib") # terms ranking model dumped in joblib

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ScoringService(object):
    model = None  # Where we keep the model when it's loaded
    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = joblib.load(model_path)
        return cls.model
    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf.predict_proba(input)[:, 1]
# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here
    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")
@app.route("/invocations", methods=["POST", "GET"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    model_columns = ['term_length', 'monthly_payment', 'income', 'application_details.modified_amount',
     'crs.MLDerogatorySummary.InquiryCount','application_details.amount_cap', 'application_details.requested_amount', 'crs.MLDerogatorySummary.RevolvingCreditUtilization',
     'crs.TradeSummary.RevolvingHighCredit','crs.TradeSummary.RevolvingBalance', 'provider_address_id',  'crs.MLDerogatorySummary.TotalUnsecuredLoanBalance','crs.TradeSummary.TotalLiabilityPayment', 'crs.TradeSummary.RevolvingPayment', 
     'crs.TradeSummary.AutoPayment', 'crs.TradeSummary.OpenHighCredit', 
     'crs.TradeSummary.EducationCount', 'application_details.subsidy', 'crs.MLDerogatorySummary.DisputeCount']
   
    model_dtypes = {
        'term_length': 'int64',
        'income': 'float64',
        'monthly_payment': 'float64',
        'application_details.modified_amount': 'float64',
        'crs.MLDerogatorySummary.InquiryCount': 'float64',
        'application_details.amount_cap': 'float64',
        'application_details.requested_amount': 'float64',
        'crs.MLDerogatorySummary.RevolvingCreditUtilization': 'float64',
        'crs.MLDerogatorySummary.RevolvingHighCredit': 'float64',
        'crs.TradeSummary.RevolvingBalance': 'float64',
        'provider_address_id': 'float64',
        'crs.MLDerogatorySummary.TotalUnsecuredLoanBalance': 'float64',
        'crs.TradeSummary.TotalLiabilityPayment': 'float64',
        'crs.TradeSummary.RevolvingPayment': 'float64',
        'crs.TradeSummary.AutoPayment': 'float64',
        'crs.TradeSummary.OpenHighCredit': 'float64',
        'crs.TradeSummary.EducationCount': 'float64',
        'crs.TradeSummary.RevolvingHighCredit': 'float64',
        'application_details.subsidy': 'float64',
        'crs.MLDerogatorySummary.DisputeCount': 'float64'
    }
    # Convert from CSV to pandas
    if flask.request.content_type == "application/json":
        data = flask.request.data.decode("utf-8")
        data = pd.read_json(data)
    else:
        return flask.Response(
            response="This predictor only supports JSON data", status=415, mimetype="text/plain"
        )
    data = data.astype(model_dtypes)
    data = data[model_columns]
    print("Invoked with {} records".format(data.shape[0]))
    # Do the prediction
    predictions = ScoringService.predict(data)
    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({"prob_of_acceptance": predictions}).to_json(out, orient='records')
    result = out.getvalue()
    return flask.Response(response=result, status=200, mimetype="application/json")
