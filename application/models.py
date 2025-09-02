from django.db import models

class myuser(models.Model):
    id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=50, null=True) # Assuming username is unique
    contact = models.CharField(max_length=50)
    email = models.CharField(max_length=100)  # Increased max_length assuming it's an email
    password = models.CharField(max_length=128)  # Increased max_length for password

    def __str__(self):
        return self.username  # Returning username for easier identification

class question(models.Model):
    que = models.CharField(max_length=200, null=True)
    answer = models.CharField(max_length=1000, null=True)
    uid = models.IntegerField(null=True)
    rid = models.AutoField(primary_key=True)  
    
class answer(models.Model):
    answers = models.CharField(max_length=300, null=True)
    uid = models.IntegerField(null=True)
    similarity = models.CharField(max_length=200, null=True)
    userid = models.IntegerField(null=True)
    que = models.CharField(max_length=900, null=True)