

# Create your views here.

from django.shortcuts import render
from django.http import JsonResponse
# from .rag import generate_response
from .bot import generate_response

def chat_view(request):
    print('Hello world'.center(100, '-'))
    return render(request, "chatbot/Chat3.html")

def get_response(request):
    user_query = request.GET.get("message")
    if user_query:
        bot_reply = generate_response(user_query)
        return JsonResponse({"reply": bot_reply})
    return JsonResponse({"reply": "Please ask something."})
