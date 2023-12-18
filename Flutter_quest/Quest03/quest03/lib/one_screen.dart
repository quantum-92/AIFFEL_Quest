import 'package:flutter/material.dart';
//import 'user.dart';

class CatScreen extends StatelessWidget {
  CatScreen({super.key});
  // is_cat 변수를 정의합니다.
  bool is_cat = true;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        home: Scaffold(
      // appBar에 아이콕과 타이틀을 추가합니다.
      appBar: AppBar(
        leading: const Icon(Icons.animation),
        title: const Text('First Page'),
      ),
      body: Container(
        //color: Colors.red,
        child: Center(
            child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () async {
                // is_cat = false로 변경하고, /dog로 이동한다.
                //
                print("is_cat: $is_cat");
                is_cat = false;
                //final result = await Navigator.pushNamed(context, '/two',
                Navigator.pushNamed(context, '/dog',
                    arguments: {"arg1": is_cat});
                //print('result:${(result as User).name}');
                is_cat = true;
              },
              child: const Text('Next'),
            ),
            //Image.asset('images/cat.jpg')
            Image.network(
                "https://pic3.zhimg.com/v2-1f8c53485821ed50a27bc123faa7c25a_r.jpg",
                width: 300,
                height: 300,
                fit: BoxFit.cover)
          ],
        )),
      ),
    ));
  }
}
