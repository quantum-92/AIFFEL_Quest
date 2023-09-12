# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 이혁희
- 리뷰어 : 오선우


# PRT(Peer Review Template)

- [O]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    

 	#  단어를 입력 받는다.
        text = str(input(('입력값:  \n')))

        #단어를 뒤집어서 출력한다.
        def reverse_text(text):

        # 텍스트의 길이를 구한다.
        text_len = len(text)

  	# 텍스트 스트링를 리스트로 변환한다.
  	# (스트링은 특정 인덱스의 텍스트 업데이트가 안됨)
  	text_list = list(text)

  	# 리스트를 뒤집는다.
  	for i in range(int(text_len / 2)):
    	text_list[i], text_list[- (i + 1)] = text_list[- (i + 1)], text_list[i]

  	# 리스트를 다시 스트링으로 변환
  	reversed_text = ""
  	for c in text_list:
      	reversed_text += c

  	return reversed_text

	eversed_text = reverse_text(text)

	print("뒤집힌 단어는:", reversed_text)
	
       ```
        # 뒤집은 단어가 원래의 단어와 같은지 여부를 출력한다.
   	if text.replace(' ','') == reversed_text.replace(' ',''):
  	  print("입력된 단어는 회문입니다.")
	else:
 	  print("입력된 단어는 회문이 아닙니다.")```
       
- [O]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
     잘 이해 되었습니다.
            
- [O]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 스트링은 특정 인덱스의 텍스트 업데이트가 안되었던 부분에 대해 해결을 했었던 것 같다. 
    - 결과체크를 위해 구글에서 회문문장 예시를 가져와 넣어봤는데, 띄어쓰기 부분에서 회문문장 여부가 불일치 했다.  
        
- [O]  **4. 회고를 잘 작성했나요?**
    - 회고를 어디서 확인하는건지 모르겠습니다. ;; 
    
- [O]  **5. 코드가 간결하고 효율적인가요?**
    - 스트링은 특정 인덱스의 텍스트 업데이트가 되지 않아 반복문을 사용한 부분에서 다소 복잡해졌을 것 같습니다. 생각지도 못했던 다양한 방법과 시각을 알수있어서 좋았습니다. 감사합니다. 
    

# 참고 링크 및 코드 개선
  https://blockdmask.tistory.com/568    
if text.replace(' ','') == reversed_text.replace(' ',''):

