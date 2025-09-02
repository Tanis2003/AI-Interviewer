"""Recommendationsystem URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import index
##from . import UserDashboard
##from . import index
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls), 
    path('',index.login,name="login"),
    path('logout/',index.logout,name="logout"),
    path('myregister/', index.register,name="register"),
    
    path('aboutus/',index.about,name="about"),
    path('myprofile/',index.myprofile,name="myprofile"),
    path('inputquestions/',index.inputquestions,name="inputquestions"),
    path('showquestion/',index.showquestion,name="showquestion"),
    path('service/',index.service,name="service"),
    path('doregister/',index.doregister,name="doregister"),
    path('dologin/',index.dologin,name="dologin"),
    path('viewuserprofile/',index.viewuser,name="viewuser"),
    path('viewpredicadmin/',index.viewpredicadmin,name="viewpredicadmin"),
    path('chatbot/',index.chatbot,name="chatbot"),
    path('livepred/',index.livepred,name="livepred"),
    path('prevpred/',index.prevpred,name="prevpred"),
    path('dashboard/',index.dashboard,name="dashboard"),
    path('UserDashboard/',index.UserDashboard,name="UserDashboard"),
    path('questions/',index.questions,name="questions"),
    path('chat/',index.chat,name="chat"),
    path('chatbot1/',index.chatbot1,name="chatbot1"),
    path('question_display/',index.question_display,name="question_display"),
    path('c_question_display/',index.c_question_display,name="c_question_display"),
    path('delete_question/',index.delete_question,name="delete_question"),
    path('delete_user/',index.delete_user,name="delete_user"),
    path('edit_profile/',index.edit_profile,name="edit_profile"),

    
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

