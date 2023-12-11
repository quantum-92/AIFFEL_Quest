# 회고

### 잘한 점
- 실행되는 코드를 작성하였다.

### 문제점
- Peer coder와 공동 작업을 잘 하지 못하였다.

### 배운 점
- 

### 부록
- 아래는 내가 작성한 코드인데 GPT의 도움을 많이 받기도 했고 해서 기록용으로만 남겨 둔다.
- 생성자를 이용해서 workingTime, breakTime을 다르게 적용할 수 있게 하였다.
- 생성자를 이용하다 보니 한번 돌 때마다 오브젝트를 새로 생성해야 하는데 느낌상 좀 이상하다.
- restart(workingTimeMin, breakTimeMin)와 같은 함수를 만들면 좀더 깔끔할 듯

```
import 'dart:async';

class PomodoroTimer {
  int workTime = 25 * 60; // 25 minutes in seconds
  int breakTime = 5 * 60; // 5 minutes in seconds

  int _remainingTime = 0;
  bool _isWorkTime = true;
  Timer? _timer;

  PomodoroTimer(int workTimeMin, int breakTimeMin) {
    workTime = workTimeMin * 60;
    breakTime = breakTimeMin * 60;
    _remainingTime = workTime;
  }

  void start() {
    print("Pomodoro Timer를 시작합니다.");
    _timer = Timer.periodic(Duration(seconds: 1), (timer) {
      _tick();
    });
  }

  void _tick() {
    _remainingTime--;
    if (_remainingTime <= 0) {
      if (_isWorkTime) {
        _isWorkTime = false;
        _remainingTime = breakTime;
        print("작업시간이 종료되었습니다. 휴식시간을 시작합니다.");
      } else {
        print("휴식시간이 종료되었습니다.");
        stop();
      }
    } else {
      print('남은 시간: ${_formatTime(_remainingTime)}');
    }
  }

  String _formatTime(int seconds) {
    int minutes = seconds ~/ 60;
    int remainingSeconds = seconds % 60;
    return '${minutes.toString().padLeft(2, '0')}:${remainingSeconds.toString().padLeft(2, '0')}';
  }

  void stop() {
    _timer?.cancel();
  }

  void reset() {
    _isWorkTime = true;
    _remainingTime = workTime;
    _timer?.cancel();
  }
}

void main() {
  PomodoroTimer timer;

    var timeSettings = [
    {'workTime': 25, 'breakTime': 5},
    {'workTime': 25, 'breakTime': 5},
    {'workTime': 25, 'breakTime': 5},
    {'workTime': 15, 'breakTime': 15},
  ];

  for (var setting in timeSettings) {
    timer = PomodoroTimer(setting['workTime']??0, setting['breakTime']??0);
    timer.start();
  }
}

```