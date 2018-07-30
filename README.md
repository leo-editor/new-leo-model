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

# Why new data model

Currently Leo uses VNode instances to hold all outline data and Position instances to traverse
and operate on the outline. VNode instances are linked directly to one another which allows
modification of outline in O(1) constant time. But this convenience comes with the price. It
makes very hard to search for any given node. It is like a database table without indexes
(insert and delete operations are cheap, but searching is expensive). For example to extract
visible nodes from this structure one has to search from the top going from one node to another
without shortcuts. This makes drawing operations very expensive, but not only drawing. Implementing
find and replace commands is also very hard. Searching tree for '@path' and other directives
is also more expensive than necessary. Undo/redo functions are also hard to implement.

My primary goal was to make a data model that will keep its data under controll allowing
modifications only through its methods. This allows keeping some kind of indexes in database
which can speed up queries and reduce necessary work during searches. Keeping indexes comes
also with price. It requires additional work when modifying outline, but the experiments
clearly show that this additional work is far less than the work Leo currently does while
traversing and searching. After all modifications are far less frequent than searches and
traversals. Each time tree is redrawn or saved, or loaded or analyzed traversals and searches
are executed.

The other reason for making new data model is that current classes VNode and Positions can't
be instantiated without commander instance which can't exists without VNode-s and Position-s.
It is closed loop that makes testing and using code harder than necessary. Initialization code
is very hard to reason about and understand in part due to this interdependence of Commands,
VNode and Position classes. I wished data model that can be used with or without any gui, that
can be tested with simple input data that consists of integers, floats, strings, tuples,
dictionaries and lists. If, for testing, you don't need to provide any more complex input data
then it is possible to write more test cases and to cover code with tests more thoroughly.

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
```

To install hypothesis you have to use version from github and not pypi. Simply go to 
[releases page](https://github.com/HypothesisWorks/hypothesis/releases), download and unpack
latest release and inside unpacked source execute:

```
cd hypothesis-python
python setup.py install
```

A good way to learn about how model works is to comment out any line in code and run
tests. Most likely test will fail and show you what sequence of tree operations lead
to failure. As a convinience every time new outline is generated and tested it is also
saved in `tmp/trdata001.bin` and sequence of operations is dumped in the console. It
is easy to copy and paste this sequence in test function which would reproduce the
same failure when executed. Then it is possible to generate svg images of the outline
state in this process and look at each step and find why test fails.

