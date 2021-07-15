from django.shortcuts import render
from data_app.models import Algorithm, DataModel
from SyntheticDataGeneration import *
from BordaCount import *
from RankCentrality import *
from LeastSquaresAlgorithm import *
from BTL_Model import *
import csv
from django.core.files.storage import FileSystemStorage
from django.conf import settings
# Create your views here.
def index(request):
    data = Algorithm.objects.all()
    data1 = DataModel.objects.all()
    algol = {
    "algorithmname": data,
    "modelname": data1,
    }
    score=[]
    flag=0
    error=0
    alg=""
    acu=0
    data=""
    model_file=""
    final_data=""
    #if request.method == 'POST' and 'Train' in request.POST:
    if request.method == 'POST' and 'Train' in request.POST:
        dict1=request.POST
        list = request.POST.getlist('algorithm')
        with open('data1.csv','w') as csvfile:
            wrt=csv.writer(csvfile)
            for key, value in dict1.items():
                wrt.writerow([key,value])
        if dict1['dataset_type'] == 'Synthetic':
            if dict1['model'] == 'Exact P'  and request.FILES['myfile']:
                os.remove(os.path.join(settings.MEDIA_ROOT, 'Model.csv'))
                myfile = request.FILES['myfile']
                fs = FileSystemStorage()
                filename = fs.save('Model.csv', myfile)
                dataset = pd.read_csv('media/Model.csv',sep=',')
                exactP = dataset.iloc[:].values
                noItems=(len(exactP))
                noComparison=(dict1['pairs'])
                frequency=(dict1['repetitions'])
                synthetic_data(noItems,noComparison,frequency)
                data = "media/Syntheticdataset.csv"
                flag=1
                model_file = "media/Model.csv"

            elif dict1['model'] == 'BTL Model':
                noItems=(dict1['items'])
                noComparison=(dict1['pairs'])
                frequency=(dict1['repetitions'])
                BTLsynthetic_data(noItems,noComparison,frequency)
                data = "media/Syntheticdataset.csv"
                model_file = "media/Model.csv"
                flag=1

        final_data={}
        if 'Borda Count' in list:
            alg="Borda Count"
            data = "media/Syntheticdataset.csv"
            score,error,acu=borda_count(data,dict1['train'],noItems,flag)
            #bc={"alg":Borda Count,"sc":score,"er":error,"ac":acc}
            final_data["bc"]={"alg":"Borda Count","sc":score,"er":error,"ac":acu}

        if 'Rank Centrality' in list:
            alg="Rank Centrality"
            data = "media/Syntheticdataset.csv"
            score,error,acu=rank_centrality(data,dict1['train'],noItems)
            #rc={"alg":Rank Centrality,"sc":score,"er":error,"ac":acc}
            final_data["rc"]={"alg":"Rank Centrality","sc":score,"er":error,"ac":acu}
        if 'Least Square' in list:
            alg="Least Square"
            data = "media/Syntheticdataset.csv"
            score,error,acu=least_squares(data,dict1['train'],noItems)
            #rc={"alg":Rank Centrality,"sc":score,"er":error,"ac":acc}
            final_data["lc"]={"alg":"Least Square","sc":score,"er":error,"ac":acu}


    #return render(request,"data_app/index.html",context = {"db":algol,"model":model_file, "file":data,
    #"sigma":score, "res_error":error,"algm":alg,"accuracy":acu})

    return render(request,"data_app/index.html",context = {"db":algol,"model":model_file, "file":data,
    "dat":final_data})


def adminpage(request):
    data = Algorithm.objects.all()
    data1 = DataModel.objects.all()
    algol = {
    "algorithmname": data,
    "modelname": data1
    }
    return render(request,"data_app/adminpage.html",context=algol)
