# Main Quest README

- 작성자 : 이혁희

## 개요
1. 이 문서는 Main Quest 코드에 대한 설명입니다. 
2. 프로그램은 Accoun.ipynb와 Aiffel.ipynb로 되어 있고 셀을 위에서 아래로 순차적으로 실행하도록 되어 있습니다.

## 메인퀘스트 1번 : Account
1. 은행계좌의 생성, 입금, 출금, 100만원 넘는 계좌 정보 출력합니다.
2. 마지막 셀에 다시 계좌를 생성하고 입출력을 반복한 후에 입금내역과 출금내역을 출력합니다.
3. 입금 5회까지 이자가 지급되는 것을 보여주기 위해서 6번 입금하였습니다.
   ```
   # 입출금 반복. 
   # 5회까지 이자가 지급되는 것을 확인하기 위하여 입금 6회 반복
   a.deposit(500000)
   a.withdraw(1000000)
   a.deposit(100000)
   a.deposit(100000)
   a.deposit(100000)
   a.deposit(100000)
   a.deposit(100000)
   a.withdraw(500000)
   ```

## 메인퀘스트 2번 : Aiffel
1. 순서대로 Aiffel, Aiffel_Guild, Aiffel_Group 클래스를 정의하고 필요한 테스트를 진행했습니다.
2. 본 퀘스트에서 grews 데이터의 key는 'name'이라고 가정했습니다. 즉, 이름이 동일한 사람이 없다는 가정입니다.   
1. Aiffel클래스의 __grews__ 변수는 그루들의 모든 정보를 저장하는 DataFrame입니다.
   자식 클래스에서도 사용할 수 있도록 클래스 변수로 정의하였습니다. 문제에서 Aiffel_Group, Aiffel_Guild 클래스에서  Aiffel 클래스를 상속 받아 처리하게 되어 있는데 주요 데이터를 개체들 간에 공유할 방법이 class 변수로 만드는 것 외에는 생각나지 않았습니다.
3. Aiffel_Group.group2guild_score함수에서 Aiffel_Guild.score함수를 콜해서 그룹에 속한 사람이 속한 길드의 점수를 업데이트 했습니다.
   그런데, guild_score함수를 콜하기 위해서는 Aiffel_Guild 개체에 대한 reference를 알고 있어야 하는데 상속으로 값을 공유하는 방법을 알아 내지 못했습니다.
   그래서, Aiffel_Guild 개레를 생성한 후에 Aiffel_Gourp.aiffel_guild변수에 assign해 주었습니다.
   ```
   a_group.aiffel_guild = a_guild
   ```
6. 프로그램의 실행은 다음 순서를 따릅니다.
>1. Aiffel
>>  - 초기화
>>   ```
>>   # Aiffel 개체 생성
>>   aiffel = Aiffel()
>>   # 딕셔너리로 grew 리스트를 받아 입력
>>   aiffel.add_grews(grews)
>>   ```
>>   - 그루에게 상벌점 주기
>>   - 그루 2명 삭제
>2. Aiffel_Guild
>>   - 초기화
>>   ```
>>   # Aiffel_Guild 개체 생성
>>   a_guild = Aiffel_Guild()
>>   # 각 그루를 길드에 mapping
>>   a_guild.map_guild()
>>   ```
>>   - 길드에게 상벌점 주기
>3. Aiffel_Group
>>   - 초기화
>>   ```
>>   # Aiffel_Group 개체 생성
>>   a_group = Aiffel_Group()
>>   # 각 그루에게 그룹을 매핑한다.
>>   a_group.map_group()
>>   # a_group.aiffel_guild 변수에 Aiffel_Guild개체(a_guild)를 assign한다.
>>   # group2guild_score함수에서 a_guild.guild_score를 호출하기 위함.
>>   a_group.aiffel_guild = a_guild
>>   ```
>>   - group2guild_score 실행
   
