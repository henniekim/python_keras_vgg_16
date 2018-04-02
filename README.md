# PYTHON KERAS VGG16 

본 프로젝트는 "Very deep convolutional networks for large-scale image recognition" 논문에 있는 아키텍쳐를 tensorflow와 keras를 이용하여 구현하였습니다.
논문과는 달리 ILSVRC 데이터를 사용하지 않고, PASCAL VOC 데이터만을 이용했으며, 이 때문에 성능이 좋지 않습니다. (~55%)
Fine tuning 네트워크를 이용하면 보다 향상된 정확도를 얻을 수 있습니다. (~87%)
## 설치 방법
 
## 사용 예제
터미널 창에서 다음과 같이 실행하세요.
```sh
python3 keras_vgg_16.py
```
Fine-tuning architecture를 사용하려면 다음과 같이 실행하세요.
```sh
python3 keras_vgg_16_fine_tune.py
```
## 개발 환경 설정
Python 3.x 버젼과, tensorflow, keras, opencv 가 필요합니다.
```sh
pip3 install tensorflow
pip3 install keras
```
 
## 업데이트 내역

* 0.0.1
    * Scratch training architecture 추가
* 0.0.2
    * Fine-tuning architecture 추가
 
## 정보

김동현 – seru_s@me.com

## 라이센스

MIT © henniekim

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
