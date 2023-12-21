### AIFFEL Campus Online Code Peer Review Templete

- 코더 : 이혁희
- 리뷰어 : 김연 

### PRT

- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 퀘스트 문제 요구조건 등을 지칭
    - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부

```
Future<void> fetchData(int queryType) async {
    try {
      const enteredUrl = "https://c5db-34-82-13-125.ngrok-free.app/";
      //"https://57f0-34-82-13-125.ngrok-free.app/"; // 입력된 URL 가져오기
      final response = await http.get(
        Uri.parse("${enteredUrl}sample"), // 입력된 URL 사용
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': '69420',
        },
      );
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          if (queryType == 1) {
            result = "예측 라벨: ${data['predicted_label']}";
          } else {
            result = "예측 확률: ${data['prediction_score']}";
          }
        });
      } else {
        setState(() {
          result = "Failed to fetch data. Status Code: ${response.statusCode}";
        });
      }
      print(result);
    } catch (e) {
      setState(() {
        result = "Error: $e";
      });
    }
  }
```

> 네. 각각의 버튼을 누르면 확률과 레이블이 나타나는 미니앱을 완성하셨습니다.
      
    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인

```
ElevatedButton(
    onPressed: () => fetchData(1),
    child: const Text("예측결과"),
    ),
ElevatedButton(
    onPressed: () => fetchData(2),
    child: const Text("예측확률"),
    ),
```

> 저는 좀 복잡하게 접근했던 부분이었는데 간단하게 해결하셔서 인상적입니다. 특히 fetchData(1) 이렇게 처리하신 부분이 흥미롭습니다.

   
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나”, ”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 실험이 기록되어 있는지 확인
    - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

> 에러가 발생한 부분은 없습니다.
        
- [x]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해 배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
    - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
     
```
## 회고
### 잘한 점
- fast api를 통하여 서버의 서비스를 모바일에서 보여주는 서비스를 구현하였다.
### 문제점
- DLthon에서 학습한 모델을 서비스 모델로 사용하지 못하였다. 
### 배운 점
- 모델학습할 때 학습된 모델을 저장하는 습관을 들이자.
```

> 네. 회고를 잘 작성해주셨습니다.


- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
    - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

> 네 전체적으로 간결합니다. 수고하셨습니다!


