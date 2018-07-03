# The new-leo-model repo.
This repo contains code of proposed new data model for Leo. The so called *new
data model* is at the moment contained in just one python file [leoDataModel.py](leoDataModel.py). 
Other files in this repository are intended to demonstrate new data model in action.

Current version is the result of many experiments and prototypes I have built
in last twelve months or so (July 2017 - July 2018).

1. first experiments in [snippets/experiments](https://github.com/leo-editor/snippets/blob/master/experiments/line-numbering.leo)
2. Leo in ClojureScript [leo-cljs](https://repo.computingart.net/leocljs/home)
3. Leo in CoffeeScript [leo-el-vue](https://leoelvue.computingart.net/index)

The model has been implemented in several different languages, platforms and
versions. Each time I have built a new version, I learned something new and improved
the model a bit.

If you want to know how new model works, you can find some explanations and examples
in my [blog](https://computingart.net).

# Testing new model
For testing this project uses excellent [hypothesis](https://hypothesis.works) framework.
The testing process involves generating pseudo random outlines and performing some
modifications in pseudo random order, checking model data is always consistent. This
kind of testing is very powerfull way to test model. Several bugs vere discovered and
fixed using this process. Most of them would be almost impossible to come up with the
such test case, and some of them no one would ever encounter in normal use of model.
But, thanks to the [hypothesis](https://hypothesis.works), these bugs were exterminated
even before anyone has seen them.

To test model just run:

```
pytest test_ltm.py
```

Of course you need to install `pytest` and `hypothesis` before:

```
pip install pytest
pip install hypothesis
```

A good way to learn about how model works is to comment out any line in code and run
tests. Most likely test will fail and show you what sequence of tree operations lead
to failure. As a convinience every time new outline is generated and tested it is also
saved in `tmp/trdata001.bin` and sequence of operations is dumped in the console. It
is easy to copy and paste this sequence in test function which would reproduce the
same failure when executed. Then it is possible to generate svg images of the outline
state in this process and look at each step and find why test fails.

