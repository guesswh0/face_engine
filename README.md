# FaceEngine

Project main purpose is to simplify work with __face recognition problem__ 
computation core trio - *detector*, *embedder*, and *predictor*. FaceEngine 
combines all of them in one interface model to simplify usage and furthermore 
extends facilities and adds some features.

## Usage

#### Getting it
To download FaceEngine, either pull this github repo or simply use Pypi via pip:

    $ pip3 install face-engine
FaceEngine is supported only on Python 3.6 and above.
    
To fetch project default models use: 

    $ fetch_models

#### Using it:
FaceEngine is working out of the box, with pre-defined default models:

```python
>>> from face_engine import FaceEngine
>>> engine = FaceEngine()
```
to change default models use appropriate setter methods, for example to use 
more robust face detector model `'mmod'` (see [`face_engine/models/mmod_detector.py`](https://github.com/guesswh0/face_engine/blob/master/face_engine/models/mmod_detector.py)
) use:

```python
>>> engine.detector = 'mmod'
```
 
 
#### Lets do some __face recognition__:

1. prepare some dataset with image urls/paths and class_names
 
```python
>>> images = ['person1.jpg', 'person2.jpg', 'person3.jpg']
>>> class_names = ['person1', 'person2', 'person3']
```

2. fit predictor model with prepared data

```python
>>> engine.fit(images, class_names)
```

3. and finally make predictions on test images

```python
>>> from skimage import io
>>> image = io.imread('test_image.jpg.')
>>> scores, class_names = engine.predict(image)
```

### Custom models
Pre-defined default models are used to show how to work with FaceEngine. 
These models are working pretty good, but if you want to, you can work with your 
own pre-trained models. FaceEngine is designed the way, when you can easily 
plugin your own model. All you need to do is to implement model interface 
`Detector`, `Embedder` or `Predictor` (see [models](https://github.com/guesswh0/face_engine/blob/master/face_engine/models/__init__.py) 
package), and import it with either directly importing your model or adding it 
to `PYTHONPATH` environment variable or using appropriate convenient function 
`from face_engine.tools`. This will <ins>register</ins> your model class object itself 
in `models` dictionary, from where it become visible.

To init with your own pre-trained detector use:
```python
>>> from my_custom_models import my_custom_detector
>>> engine = FaceEngine(detector='custom_detector')
```

To switch to your own model use corresponding setter method:

```python
>>> from my_custom_models import my_custom_detector
>>> engine.detector = 'custom_detector'
```
 
How to train your own custom model is out of this user guide scope =).

## Notice
There is also a few methods, but it is better if you will try to figure them 
out by yourself.

I didn't wrote full documentation or tutorial yet (hope doing so sooner or later), 
in the meantime for more detailed info on models or engine itself see docstrings.

Questions? Issues? Feel free to [ask](https://github.com/guesswh0/face_engine/issues/new).
## License

Apache License Version 2.0

Copyright 2019 Daniyar Kussainov

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.