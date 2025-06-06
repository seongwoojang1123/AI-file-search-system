import json
import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import AccessLog, Feedback

FASTAPI_URLS = {
    "keyword": "http://50.201.102.69:8002/ask/",
    "title": "http://50.201.102.69:8002/ask2/",
    "title-keyword": "http://50.201.102.69:8002/ask3/"
}
SUMMARY_URL = "http://50.201.102.69:8002/summary/"

def chat_page(request):
    n_results_options = list(range(3, 21))
    return render(request, "chat.html", {"n_results_options": n_results_options})

@csrf_exempt
def ai_chat(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "").strip()
            n_results = str(data.get("n_results", 3))
            tab = request.GET.get("tab", "keyword")  # 기본값: keyword

            if not user_message:
                return JsonResponse({"error": "질문이 비어 있습니다."}, status=400)

            client_ip = request.META.get('REMOTE_ADDR')
            access_log = AccessLog.objects.create(client_ip=client_ip, question=user_message)

            headers = {"Content-Type": "application/json"}
            payload = {"question": user_message, "n_results": n_results}
            fastapi_url = FASTAPI_URLS.get(tab, FASTAPI_URLS["keyword"])
            response = requests.post(fastapi_url, json=payload, headers=headers, timeout=240)

            if response.status_code != 200:
                return JsonResponse({
                    "status": "error",
                    "message": f"FastAPI 응답 오류: {response.status_code}"
                }, status=response.status_code)

            response_data = response.json()
            filtered_results = []
            expected_n_results = int(n_results)

            if isinstance(response_data, dict):
                names = response_data.get("name", [])
                paths = response_data.get("path", [])
                texts = response_data.get("text", [])
                for i in range(min(len(names), len(paths), expected_n_results)):
                    name = names[i] if i < len(names) else f"unnamed_{i+1}"
                    path = paths[i] if i < len(paths) else f"/unknown/path_{i+1}"
                    text = texts[i] if i < len(texts) else "내용 없음"
                    filtered_results.append({"name": name, "path": path, "text": text})

            if len(filtered_results) < expected_n_results:
                while len(filtered_results) < expected_n_results:
                    filtered_results.append({
                        "name": f"dummy_{len(filtered_results) + 1}",
                        "path": f"/dummy/path_{len(filtered_results) + 1}",
                        "text": "더미 내용"
                    })

            paths_list = [result["path"] for result in filtered_results]
            paths_str = "\n".join(paths_list)

            return JsonResponse({
                "status": "success",
                "message": "AI 처리 결과를 받았습니다.",
                "results": filtered_results,
                "access_log_id": access_log.id,
                "paths": paths_str,
                "question": user_message,
                "ip": client_ip
            }, status=200)

        except Exception as e:
            print(f"서버 오류: {str(e)}")
            return JsonResponse({"error": "서버 내부 오류가 발생했습니다."}, status=500)

    return JsonResponse({"error": "잘못된 요청 방식입니다. POST를 사용하세요."}, status=405)

@csrf_exempt
def navigate(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            path = data.get("path", "").strip()
            if not path:
                return JsonResponse({"error": "경로가 비어 있습니다."}, status=400)
            return JsonResponse({"status": "success", "path": path, "message": "경로를 팝업으로 표시합니다."})
        except Exception as e:
            return JsonResponse({"error": f"경로 처리 실패: {str(e)}"}, status=500)
    return JsonResponse({"error": "잘못된 요청 방식입니다. POST를 사용하세요."}, status=405)

@csrf_exempt
def summarize(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            text = data.get("text", "").strip()
            question = data.get("question", "").strip()
            if not text:
                return JsonResponse({"error": "텍스트가 비어 있습니다."}, status=400)

            headers = {"Content-Type": "application/json"}
            payload = {"text": text}
            payload["question"] = question
            response = requests.post(SUMMARY_URL, json=payload, headers=headers, timeout=240)

            if response.status_code != 200:
                error_detail = response.json().get("detail", "알 수 없는 오류")
                return JsonResponse({
                    "status": "error",
                    "message": f"요약 요청 오류: {error_detail}"
                }, status=response.status_code)

            response_data = response.json()
            return JsonResponse({"status": "success", "answer": response_data.get("answer", "요약 없음")})

        except Exception as e:
            return JsonResponse({"error": f"요약 처리 실패: {str(e)}"}, status=500)
    return JsonResponse({"error": "잘못된 요청 방식입니다. POST를 사용하세요."}, status=405)

@csrf_exempt
def submit_feedback(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            access_log_id = data.get("access_log_id")
            rating = data.get("rating")
            response_text = data.get("response_text")  # paths
            question = data.get("question")
            ip = data.get("ip")

            if not all([access_log_id, rating, response_text, question, ip]):
                return JsonResponse({"error": "필수 데이터 누락"}, status=400)

            access_log = AccessLog.objects.get(id=access_log_id)
            Feedback.objects.create(
                access_log=access_log,
                response_text=response_text,
                rating=rating,
                question=question,
                ip=ip
            )
            return JsonResponse({"status": "success", "message": "피드백이 저장되었습니다."})

        except AccessLog.DoesNotExist:
            return JsonResponse({"error": "해당 질문 기록을 찾을 수 없습니다."}, status=404)
        except Exception as e:
            print(f"피드백 저장 오류: {str(e)}")
            return JsonResponse({"error": f"피드백 저장 실패: {str(e)}"}, status=500)
    return JsonResponse({"error": "잘못된 요청 방식입니다. POST를 사용하세요."}, status=405)