from django.urls import path
from .views import predict_diabetes,home,breast_cancer_predict,heart_disease_predict,kidney_disease_predict,kidney_disease_view,gemini_chat

urlpatterns = [
    path("", home, name="home"),
    path("diabetes/", predict_diabetes, name="predict"),
    path("breast_cancer/", breast_cancer_predict, name="breast_cancer"),
    path("heart_disease/", heart_disease_predict, name="heart_disease"),
    path("kidney_disease/", kidney_disease_predict, name="kidney_disease_predict"),
    path('kidney_disease/', kidney_disease_view, name='kidney_disease'),
    path("gemini/", gemini_chat, name="gemini_chat"),
]