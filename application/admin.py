from django.contrib import admin
from application.models import myuser
from application.models import question
from application.models import answer

class Adminmyuser(admin.ModelAdmin):
    list_display =('id','username','contact','email','password')

admin.site.register(myuser, Adminmyuser)

class Adminquestion(admin.ModelAdmin):
    list_display =('que','answer','uid','rid')

admin.site.register(question, Adminquestion)

class Adminanswer(admin.ModelAdmin):
    list_display =('answers','uid','similarity','userid','que')

admin.site.register(answer, Adminanswer)