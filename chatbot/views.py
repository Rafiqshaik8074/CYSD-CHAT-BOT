

# Create your views here.

from django.shortcuts import render
from django.http import JsonResponse
# from .rag import generate_response
from .bot2 import generate_response

def chat_view(request):
    print('Hello world'.center(100, '-'))
    return render(request, "chatbot/Chat3.html")

# def get_response(request):
#     user_query = request.GET.get("message")
#     if user_query:
#         bot_reply = generate_response(user_query)
#         return JsonResponse({"reply": bot_reply})
#     return JsonResponse({"reply": "Please ask something."})



def get_response(request):
    try:
        user_query = request.GET.get('message', '')
        print('Message from frontend: ', user_query)
        if(user_query):
            bot_reply = generate_response(user_query)
            print('Response from ChatBot: ', bot_reply)
            return JsonResponse({'reply': bot_reply})
    except Exception as e:
        print(e)
        return JsonResponse({'error': str(e)}, status=500)
