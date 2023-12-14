import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        home: Scaffold(
            appBar: AppBar(
              leading: IconButton(
                icon: Image.asset('images/home.png'),
                //iconSize: 50, // 아이콘 크기 조절
                onPressed: () {},
              ),
              title: const Text('플러터앱 만들기'),
            ),
            body: Column(children: [
              Center(
                child: SizedBox(
                    //color: Colors.green,
                    height: 300,
                    width: 300,
                    child: Align(
                        alignment: Alignment.center,
                        //child: Container(
                        child: ElevatedButton(
                            child: const Text('버튼'),
                            onPressed: () {
                              print('Text 버튼이 눌렸습니다!!');
                            })
                        //)
                        )),
              ),
              SizedBox(
                  //color: Colors.yellow,
                  height: 300,
                  width: 300,
                  child: Stack(children: [
                    Container(
                      color: Colors.red,
                    ),
                    Container(
                      color: Colors.green,
                      height: 250,
                      width: 250,
                    ),
                    Container(
                      color: Colors.orange,
                      height: 200,
                      width: 200,
                    ),
                    Container(
                      color: Colors.blue,
                      height: 150,
                      width: 150,
                    ),
                    Container(
                      color: Colors.purple,
                      height: 100,
                      width: 100,
                    )
                  ])),
            ])));
  }
}
