# PYTHON KERAS VGG16 

본 프로젝트는 "Very deep convolutional networks for large-scale image recognition" 논문에 있는 아키텍쳐를 Keras를 이용하여 구현하였습니다.
논문과는 달리 ILSVRC 데이터를 사용하지 않고, PASCAL VOC 데이터만을 이용했으며, 이 때문에 성능의 한계가 있습니다.(~55%).
Fine tuning 네트워크를 이용하면 보다 향상된 정확도를 얻을 수 있습니다(~87%).
학습에 필요한 데이터는 https://github.com/henniekim/pascal_voc_data_parsing 에서 만들 수 있습니다.

감사합니다!

This project is Keras implementation of "Very deep convolutional networks for large-scale image recognition" paper.
I only use PASCAL VOC data, when the author of paper use ILSVRC data, which involves not very good performance(~55%).
I also add fine-tuning network, pretrained with ImageNet data.
You can get far better performance(~87%).
Anyone who want make the training data, you may visit https://github.com/henniekim/pascal_voc_data_parsing 

Thank you!

## INSTALLATION

별 다른 설치가 필요하지 않습니다.
복잡한 실행방법을 요하는 코드는 지양합니다.

It doesn't need any installation process.
I don't like any 'complicated' things.

## USAGE
터미널 창에서 다음과 같이 실행하세요.
```sh
python3 keras_vgg_16.py
```
Fine-tuning architecture를 사용하려면 다음과 같이 실행하세요.
```sh
python3 keras_vgg_16_fine_tune.py
```
## ENVIRONMENT
Python 3.x 버젼과, tensorflow, keras, opencv 가 필요합니다.
Pycharm IDE를 사용하였지만, 코드 실행에 특별히 필요하지는 않습니다.
```sh
pip3 install tensorflow
pip3 install keras
```
 
## UPDATE

* 0.0.1
    * Scratch training architecture 추가하였습니다.
* 0.0.2
    * Fine-tuning architecture 추가하였습니다.
* 0.0.3
    * Inference model을 추가하였습니다. 원하는 이미지를 넣으면 분류 결과를 얻을 수 있습니다.
 
## INFO

김동현 – seru_s@me.com
Donghyun Kim / Henniekim

## LICENSE

MIT © henniekim

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
