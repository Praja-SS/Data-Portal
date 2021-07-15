from django.db import models

# Create your models here.
class Algorithm (models.Model):
    algorithm_name = models.CharField(max_length=100, unique= True)
class DataModel (models.Model):
    datamodel_name = models.CharField(max_length=100, unique= True)
