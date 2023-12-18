import 'package:flutter/material.dart';
import 'one_screen.dart';
import 'two_screen.dart';

void main() {
  runApp(const MyApp());
}

// Named Route를 사용합니다.
// 2개의 화면(CatScreen, DogScreen)을 만들고, 각각의 화면에서 다른 화면으로 이동합니다.
class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      initialRoute: '/cat',
      routes: {
        '/cat': (context) => CatScreen(),
        '/dog': (context) => DogScreen(),
      },
    );
  }
}
