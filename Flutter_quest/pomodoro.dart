import 'dart:async';

class PomodoroTimer {
  int CycleCompleted = 0;

  void WorkingTime() {
    var counter = 1500;
    Timer.periodic(Duration(seconds: 1), (Timer timer) {
      if (counter > 0) {
        var minutes = counter ~/ 60;
        var seconds = counter % 60;
        print('flutter: ${minutes}m ${seconds}s');
        counter--;
      } else {
        timer.cancel();
        CycleCompleted++;
        print('작업 시간이 종료되었습니다. 휴식 시간을 시작합니다.');

        if (CycleCompleted % 4 == 0) {
          BreakingTime(900);
        } else {
          BreakingTime(300);
        }
      }
    });
  }

  void BreakingTime(int breakDuration) {
    Timer.periodic(Duration(seconds: 1), (Timer timer) {
      if (breakDuration > 0) {
        var minutes = breakDuration ~/ 60;
        var seconds = breakDuration % 60;
        print('flutter: ${minutes}m ${seconds}s');
        breakDuration--;
      } else {
        timer.cancel();
        print('휴식 시간이 종료되었습니다. 작업 시간을 시작합니다.');
        WorkingTime();
      }
    });
  }
}

void main() {
  print('포모도로 타이머를 시작합니다.');

  var pomodorotimer = PomodoroTimer();
  pomodorotimer.WorkingTime();
}
