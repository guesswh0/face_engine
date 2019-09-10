# FaceEngine - The simplest face recognition!

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
from face_engine import FaceEngine
engine = FaceEngine()
```
to change default model use appropriate setter method, for example to use 
more robust face detector model `'mmod'` (see [`face_engine/models/mmod_detector.py`](https://github.com/guesswh0/face_engine/blob/master/face_engine/models/mmod_detector.py)
) use:

```python
engine.detector = 'mmod'
```
 
 
#### Lets do some __face recognition__:

1. prepare some dataset with image urls/paths and class_names
 
    ```python
    images = ['person1.jpg', 'person2.jpg', 'person3.jpg']
    class_names = ['person1', 'person2', 'person3']
    ```

2. fit predictor model with prepared data

    ```python
    engine.fit(images, class_names)
    ```

3. and finally make predictions on test images

    ```python
    from skimage import io
    image = io.imread('test_image.jpg.')
    score, class_name = engine.predict(image)
    ```

Pre-defined default models are used to show how to work with FaceEngine. 
These models are working pretty good, but if you are ~~computer-vision~~ 
developer you probably could think of working  with your own 
pre-trained models!? With FaceEngine you can easily plug your
model in and use it. All you need to do is to implement model interface 
`Detector`, `Embedder` or `Predictor` (see [models](https://github.com/guesswh0/face_engine/blob/master/face_engine/models/__init__.py) 
package rules), `register` model (import) and `create` instance of it with 
`use_plugin` method.

How to train your own model is out of this user guide scope, try to ask 
[@davidsandberg](https://github.com/davidsandberg) ;) 

```python
engine.use_plugin(name='mmod', filepath='face_engine/models/mmod_detector.py')
```
this will import all your *dependencies* and *register* your model class 
object itself in `models` dictionary and then will *create* instance of it.   

All pre-defined default models are also considered as plugin models :heavy_exclamation_mark:


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