<aside>
🔑 **PRT(Peer Review Template)**

코드 작성자: 이혁희
리뷰어 : 서승호

- [o]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
     - 잘 제출되었습니다. 학습과정을 진행하는 코드입니다.

```
def train(dataset, epochs, save_every):
    start = time.time()
    history = {'gen_loss':[], 'disc_loss':[], 'real_accuracy':[], 'fake_accuracy':[]}

    
    for epoch in range(epochs):
        epoch_start = time.time()
        for it, image_batch in enumerate(dataset):
            gen_loss, disc_loss, real_accuracy, fake_accuracy = train_step(image_batch)
            history['gen_loss'].append(gen_loss)
            history['disc_loss'].append(disc_loss)
            history['real_accuracy'].append(real_accuracy)
            history['fake_accuracy'].append(fake_accuracy)

            if it % 50 == 0:
                display.clear_output(wait=True)
                generate_and_save_images(generator, epoch+1, it+1, seed)
                print('Epoch {} | iter {}'.format(epoch+1, it+1))
                print('Time for epoch {} : {} sec'.format(epoch+1, int(time.time()-epoch_start)))

        if (epoch + 1) % save_every == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        display.clear_output(wait=True)
        generate_and_save_images(generator, epochs, it, seed)
        print('Time for training : {} sec'.format(int(time.time()-start)))

        draw_train_history(history, epoch)
```


- [o]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
- 주석 처리가 잘 되어있습니다. 또한 추가 설명등을 통해서 잘 이해됩니다.


- [o]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
- 새로운 추가 실험한 것들 잘 적혀있습니다.


- [o]  **4. 회고를 잘 작성했나요?**
- readme 파일을 통해 잘 작성되어 있습니다.

    
- [o]  **5. 코드가 간결하고 효율적인가요?**
- 간결하고 효율적으로 작성 되어있습니다.

      
</aside>
