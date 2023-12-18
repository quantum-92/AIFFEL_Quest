import 'package:flutter/material.dart';
//import 'user.dart';

//생략...................
class DogScreen extends StatelessWidget {
  DogScreen({super.key});

  bool is_cat = false;

  @override
  Widget build(BuildContext context) {
    Map<String, Object> args =
        ModalRoute.of(context)?.settings.arguments as Map<String, Object>;
    return MaterialApp(
        home: Scaffold(
      appBar: AppBar(
        leading: const Icon(Icons.catching_pokemon),
        title: const Text('Second Screen'),
      ),
      body: Container(
        color: Colors.green,
        child: Center(
            child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () {
                print("is_cat: ${args['arg1'].toString()}");
                Navigator.pop(context);
              },
              child: const Text('Back'),
            ),
            Image.network(
                "https://img.animalplanet.co.kr/thumbnail/2020/06/10/2000/7sool8s03m92a204443j.jpg",
                width: 300,
                height: 300,
                fit: BoxFit.cover)
          ],
        )),
      ),
    ));
  }
}
