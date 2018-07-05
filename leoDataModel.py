#@+leo-ver=5-thin
#@+node:vitalije.20180510153405.1: * @file leoDataModel.py
import random as _mrandom
from collections import defaultdict, namedtuple
import re
import os
import xml.etree.ElementTree as ET
import pickle
import time
random = _mrandom.Random()
random.seed(12)
class myrandom:
    def __init__(self):
        self.x = 1
    def random(self):
        self.x += 1
        return 1 - 1/self.x
random = myrandom()
#@+others
#@+node:vitalije.20180510153747.1: ** parPosIter
def parPosIter(ps, levs):
    '''Helper iterator for parent positions. Given sequence of
       positions and corresponding levels it generates sequence
       of parent positions'''
    rootpos = ps[0]
    levPars = [rootpos for i in range(256)] # max depth 255 levels
    it = zip(ps, levs)
    next(it) # skip first root node which has no parent
    yield rootpos
    for p, l in it:
        levPars[l] = p
        yield levPars[l - 1]

#@+node:vitalije.20180510153733.2: ** nodes2treemodel
def nodes2treemodel(inodes, oldpositions=None):
    '''Creates LeoTreeModel from the sequence of tuples
        (gnx, h, b, level, size, parentGnxes, childrenGnxes)'''
    ltm = LeoTreeModel()
    (positions, nodes, attrs, levels, expanded, marked) = ltm.data
    if oldpositions is None:
        positions.extend(random.random() for x in inodes)
    else:
        positions.extend(oldpositions)
    levels.extend(x[3] for x in inodes)
    nodes.extend(x[0] for x in inodes)
    for pos, x in zip(positions, inodes):
        gnx, h, b, lev, sz, ps, chn = x
        if gnx not in attrs:
            attrs[gnx] = NData(h, b, ps, chn, sz[0])
    # root node must not have parents
    rgnx = nodes[0]
    del attrs[rgnx].parents[:]
    return ltm

#@+node:vitalije.20180510153733.1: ** vnode2treemodel
def vnode2treemodel(vnode):
    '''Utility convertor: converts VNode instance into
       LeoTreeModel instance'''
    def viter(v, lev0):
        s = [1]
        mnode = (v.gnx, v.h, v.b, lev0, s,
                [x.gnx for x in v.parents],
                [x.gnx for x in v.children])
        yield mnode
        for ch in v.children:
            for x in viter(ch, lev0 + 1):
                s[0] += 1
                yield x
    return nodes2treemodel(tuple(viter(vnode, 0)))
#@+node:vitalije.20180510153733.3: ** xml2treemodel
def xml2treemodel(xvroot, troot):
    '''Returns LeoTreeModel instance from vnodes and tnodes elements of xml Leo file'''
    parDict = defaultdict(list)
    hDict = {}
    bDict = dict((ch.attrib['tx'], ch.text or '') for ch in troot.getchildren())
    xDict = {}
    #@+others
    #@+node:vitalije.20180510153945.1: *3* xml viter
    def viter(xv, lev0, dumpingClone=False):
        s = [1]
        gnx = xv.attrib['t']
        if len(xv) == 0:
            # clone
            for ch in viter(xDict[gnx], lev0, True):
                yield ch
            return
        chs = [ch.attrib['t'] for ch in xv if ch.tag == 'v']
        if not dumpingClone:
            xDict[gnx] = xv
            hDict[gnx] = xv[0].text
            for ch in chs:
                parDict[ch].append(gnx)
        mnode = [gnx, hDict[gnx], bDict.get(gnx, ''), lev0, s, parDict[gnx], chs]
        yield mnode
        for ch in xv.getchildren():
            if ch.tag != 'v':continue
            for x in viter(ch, lev0 + 1, dumpingClone):
                s[0] += 1
                yield x

    #@+node:vitalije.20180510154050.1: *3* riter
    def riter():
        s = [1]
        chs = []
        yield 'hidden-root-vnode-gnx', '<hidden root vnode>','', 0, s, [], chs
        for xv in xvroot.getchildren():
            gnx = xv.attrib['t']
            chs.append(gnx)
            parDict[gnx].append('hidden-root-vnode-gnx')
            for ch in viter(xv, 1):
                s[0] += 1
                yield ch

    #@-others
    nodes = tuple(riter())
    return nodes2treemodel(nodes)

#@+node:vitalije.20180510153733.4: ** load_leo
def load_leo(fname):
    '''Loads given xml Leo file and returns LeoTreeModel instance'''
    with open(fname, 'rt') as inp:
        s = inp.read()
        xroot = ET.fromstring(s)
        vnodesEl = xroot.find('vnodes')
        tnodesEl = xroot.find('tnodes')
        return xml2treemodel(vnodesEl, tnodesEl)
#@+node:vitalije.20180518104947.1: ** load_external_files
def load_external_files(ltm, loaddir):
    mpaths = paths(ltm, loaddir)
    (positions, nodes, attrs, levels, expanded, marked) = ltm.data

    for gnx, ps in mpaths.items():
        h = attrs[gnx].h
        if h.startswith('@file '):
            ltm2 = ltm_from_derived_file(ps[0])
            gnx2 = ltm2.data.nodes[0]
            if gnx2 != gnx:
                ltm.change_gnx(gnx, gnx2)
            ltm.replace_node(ltm2)
        elif h.startswith('@auto ') and h.endswith('.py'):
            ltm2 = auto_py(gnx, ps[0])
            ltm2.data.attrs[gnx].h = h
            ltm.replace_node(ltm2)
    ltm.invalidate_visual()
#@+node:vitalije.20180518155338.1: ** load_leo_full
def load_leo_full(fname):
    '''Loads both given xml Leo file and external files.
       Returns LeoTreeModel instance'''
    ltm = load_leo(fname)
    loaddir = os.path.dirname(fname)
    loaddir = os.path.normpath(loaddir)
    loaddir = os.path.abspath(loaddir)
    load_external_files(ltm, loaddir)
    return ltm
#@+node:vitalije.20180518100350.1: ** paths
atFileNames = [
    "@auto-rst", "@auto","@asis",
    "@edit",
    "@file-asis", "@file-thin", "@file-nosent", "@file",
    "@clean", "@nosent",
    "@shadow",
    "@thin"
]
atFilePat = re.compile(r'^(%s)\s+(.+)$'%('|'.join(atFileNames)))

def paths(ltm, loaddir):
    '''Returns dict keys are gnx of each file node,
       and values are lists of absolute paths corresponding
       to the node.'''
    (positions, nodes, attrs, levels, expanded, marked) = ltm.data

    stack = [loaddir for x in range(255)]
    res = defaultdict(list)
    pat = re.compile(r'^@path\s+(.+)$', re.M)
    cdir = loaddir
    npath = lambda x: os.path.normpath(os.path.abspath(x))
    jpath = lambda x:npath(os.path.join(cdir, x))
    for p, gnx, lev in zip(positions, nodes, levels):
        if lev == 0: continue
        cdir = stack[lev - 1]
        h, b = attrs[gnx][:2]
        m = pat.search(h) or pat.search(b)
        if m:
            cdir = jpath(m.group(1))
            stack[lev] = cdir
        m = atFilePat.match(h)
        if m:
            res[gnx].append(jpath(m.group(2)))
    return res
#@+node:vitalije.20180624125820.1: ** gnx_iter
def gnx_iter(nodes, gnx, sz=1):
    i = 0
    try:
        while i < len(nodes):
            i = nodes.index(gnx, i)
            yield i
            i += sz
    except ValueError:
        pass
#@+node:vitalije.20180624130228.1: ** up_level_index
def up_level_index(levels, i):
    lev = levels[i]
    return levels.rfind(lev-1, 0, i) if lev > 1 else 0

parent_index = up_level_index
#@+node:vitalije.20180629170433.1: ** to_leo_pos
def to_leo_pos(ltmdata, pos, c):
    (positions, nodes, attrs, levels, expanded, marked) = ltmdata
    def pstack_rev(i):
        while i > 0:
            v = c.fileCommands.gnxDict[nodes[i]]
            pi = up_level_index(levels, i)
            ci = levels[pi+1:i].count(levels[i])
            yield v, ci
            i = pi
    ipos = positions.index(pos)
    pth = list(pstack_rev(ipos))
    pth.reverse()
    v, i = pth.pop()
    p = c.rootPosition()
    p.stack[:] = pth
    p.v = v
    p._childIndex = i
    return p
#@+node:vitalije.20180613161708.1: ** LTMData
LTMData = namedtuple('LTMData', 'positions nodes attrs levels expanded marked')

def copy_ltmdata(data):
    attrs = dict((gnx, nd.deepCopy()) for gnx, nd in data.attrs.items())
    return LTMData(
        data.positions[:],
        data.nodes[:],
        attrs,
        data.levels[:],
        set(data.expanded),
        set(data.marked)
    )
#@+node:vitalije.20180625191254.1: ** NData
class NData:
    fieldnames = 'h', 'b', 'parents', 'children', 'size'
    def __init__(self, h, b, parents, children, size):
        self.h = h
        self.b = b
        self.parents = parents
        self.children = children
        self.size = size

    def __getitem__(self, i):
        names = NData.fieldnames[i]
        if isinstance(names, tuple):
            return tuple(getattr(self, x) for x in names)
        else:
            return getattr(self, names)

    def __setitem__(self, i, v):
        setattr(self, NData.fieldnames[i], v)

    def copy(self):
        return NData(self.h, self.b, self.parents, self.children, self.size)

    def deepCopy(self):
        return NData(self.h, self.b, self.parents[:], self.children[:], self.size)

    def __iter__(self):
        yield self.h
        yield self.b
        yield self.parents
        yield self.children
        yield self.size
#@+node:vitalije.20180510153738.1: ** LeoTreeModel
class LeoTreeModel(object):
    '''Model representing all of Leo outline data.
       
       warning: work in progress - still doesn't contain all Leo data
       
       TODO: add support for unknownAttributes
             add support for status bits
             add support for gui view values
                    - body cursor position
                    - selected text'''
    def __init__(self):
        self.data = LTMData([], [], {}, bytearray(b''), set(), set())
        self.selectedPosition = None
        self._visible_positions_serial = 0
        self._visible_positions_last = -1
        self._visible_positions = tuple()
        self._selectedIndex = -1
        self._undostack = []
        self._undopos = -1
    #@+others
    #@+node:vitalije.20180510153738.2: *3* parents
    def parents(self, gnx):
        '''Returns list of gnxes of parents of node with given gnx'''
        a = self.data.attrs.get(gnx)
        return a.parents if a else []

    #@+node:vitalije.20180510153738.3: *3* children
    def children(self, gnx):
        '''Returns list of gnxes of children of the node with given gnx'''
        a = self.data.attrs.get(gnx)
        return a.children if a else []

    #@+node:vitalije.20180622143953.1: *3* _child_iterator
    def _child_iterator(self, i):
        (positions, nodes, attrs, levels, expanded, marked) = self.data

        pgnx = nodes[i]

        B = attrs[pgnx].size + i
        i += 1
        j = 0
        while i < B:
            gnx = nodes[i]
            yield j, i, gnx
            j += 1
            i += attrs[gnx].size
    #@+node:vitalije.20180613193835.1: *3* size
    @property
    def size(self):
        return len(self.data.positions)

    #@+node:vitalije.20180613194129.1: *3* body
    def body(self, gnx):
        return self.data.attrs[gnx].b
    #@+node:vitalije.20180613194137.1: *3* head
    def head(self, gnx):
        return self.data.attrs[gnx].h
    #@+node:vitalije.20180620135448.1: *3* isClone
    def isClone(self, gnx):
        return len(self.data.attrs[gnx].parents) > 1
    #@+node:vitalije.20180516103839.1: *3* selectedIndex
    @property
    def selectedIndex(self):
        p = self.selectedPosition
        if p == self.data.positions[self._selectedIndex]:
            return self._selectedIndex
        try:
            i = self.data.positions.index(p)
            self._selectedIndex = i
        except ValueError:
            i = -1
        return i
    #@+node:vitalije.20180613192634.1: *3* selectedGnx
    @property
    def selectedGnx(self):
        i = self.selectedIndex
        return self.data.nodes[i]
    #@+node:vitalije.20180613192640.1: *3* selectedBody
    @property
    def selectedBody(self):
        i = self.selectedIndex
        if i < 0:
            b = ''
        else:
            gnx = self.data.nodes[i]
            b = self.data.attrs[gnx].b
        return b
    #@+node:vitalije.20180613192643.1: *3* selectedHead
    @property
    def selectedHead(self):
        i = self.selectedIndex
        if i < 0:
            h = ''
        else:
            gnx = self.data.nodes[i]
            h = self.data.attrs[gnx].h
        return h
    #@+node:vitalije.20180516160109.1: *3* insert_leaf
    def insert_leaf(self, pos, gnx, h, b):
        i = self.data.positions.index(pos)
        return self.insert_leaf_i(i, gnx, h, b)

    def insert_leaf_i(self, i, gnx, h, b):
        (positions, nodes, attrs, levels, expanded, marked) = self.data
        pi = parent_index(levels, i)
        pp = positions[pi]
        di0 = i - pi
        pgnx = nodes[pi]
        psz = attrs[pgnx].size
        lev = levels[pi] + 1
        for pxi in gnx_iter(nodes, pgnx, psz + 1):
            j = pxi + di0
            levels.insert(j, levels[pxi] + 1)
            positions.insert(j, random.random())
            if positions[pxi] == pp:
                retp = positions[j]
            nodes.insert(j, gnx)
        if gnx in attrs:
            attrs[gnx].parents.append(pgnx)
        else:
            attrs[gnx] = NData(h, b, [pgnx], [], 1)

        update_size(attrs, pgnx, 1)
        self._update_children(pgnx)
        return retp
    #@+node:vitalije.20180616125730.1: *3* clone_node
    def clone_node(self, pos):
        (positions, nodes, attrs, levels, expanded, marked) = self.data
        if pos == positions[0]: return

        # this node
        i = positions.index(pos)
        gnx = nodes[i]
        sz0 = attrs[gnx].size
        lev0 = levels[i]
        levs = [x - lev0 for x in levels[i:i+sz0]]

        # parent
        pi = parent_index(levels, i)
        pp = positions[pi]
        pgnx = nodes[pi]
        psz = attrs[pgnx].size

        # distance
        di0 = i - pi
        di1 = di0 + sz0

        update_size(attrs, pgnx, sz0)
        for pxi in gnx_iter(nodes, pgnx, psz + sz0):
            A = pxi + di0
            B = pxi + di1
            positions[B:B] = (random.random() for x in range(sz0))
            nodes[B:B] = nodes[A:A+sz0]
            levels[B:B] = levels[A:A+sz0]

        attrs[gnx].parents.append(pgnx)
        self._update_children(pgnx)
    #@+node:vitalije.20180616151338.1: *3* _update_children
    def _update_children(self, pgnx):
        (positions, nodes, attrs, levels, expanded, marked) = self.data

        i = nodes.index(pgnx)
        chn = []
        B = i + attrs[pgnx].size
        j = i + 1
        while j < B:
            cgnx = nodes[j]
            chn.append(cgnx)
            j += attrs[cgnx].size
        attrs[pgnx].children = chn
    #@+node:vitalije.20180531210047.1: *3* change_gnx
    def change_gnx(self, gnx1, gnx2):
        if gnx2 in self.data.attrs:
            raise ValueError('Gnx:%r is already in the outline'%gnx2)
        ( positions, nodes, attrs, levels, expanded, marked) = self.data

        def swap_in(ar):
            for i, x in enumerate(ar):
                if x == gnx1:
                    ar[i] = gnx2


        attrs[gnx2] = a = attrs.pop(gnx1)
        for cgnx in a.children:
            swap_in(attrs[cgnx].parents)
        for pgnx in a.parents:
            swap_in(attrs[pgnx].children)
        if gnx1 in marked:
            marked.remove(gnx1)
            marked.add(gnx2)
        for pxi in gnx_iter(nodes, gnx1, a.size):
            nodes[pxi] = gnx2
    #@+node:vitalije.20180510194736.1: *3* replace_node
    def replace_node(self, t2):
        '''Replaces node with given subtree. This outline must contain
           node with the same gnx as root gnx of t2.'''
        (positions1, nodes1, attrs1, levels1, expanded1, marked1) = self.data
        (positions2, nodes2, attrs2, levels2, expanded2, marked2) = t2.data

        gnx = nodes2[0]
        sz0 = attrs1[gnx][4]

        i = nodes1.index(gnx)
        ns = nodes1[i+1:i+sz0]

        for x in attrs1[gnx][3]:
            attrs1[x][2] = [x1 for x1 in attrs1[x][2] if x1 != gnx]

        # this function replaces one instance of given node
        def insOne(i):
            l0 = levels1[i]
            npos = [random.random() for x in nodes2]
            npos[0] = positions1[i]
            positions1[i:i+sz0] = npos
            nodes1[i:i+sz0] = nodes2
            levels1[i:i+sz0] = [(l0 + x) for x in levels2]


        # difference in sizes between old node and new node
        dsz = len(positions2) - sz0

        # parents of this node must be preserved
        attrs2[gnx][2] = attrs1[gnx][2]

        for i in gnx_iter(nodes1, gnx, len(nodes2)):
            insOne(i)

        # some of nodes in t2 may be clones of nodes in t1
        # they will have some parents that are outside t2
        # therefore it is necessary for these nodes in t2 to
        # update their parents list by adding only those parents
        # that are not part of t2.
        t2gnx = set(nodes2)
        for x in t2gnx:
            if x not in attrs1:continue
            if x is nodes2[0]:continue
            ps = attrs1[x][2]
            attrs2[x][2].extend([y for y in ps if y not in t2gnx])

        # now we can safely update attrs dict of t1
        attrs1.update(attrs2)

        # one last task is to update size in all ancestors of replaced node
        def updateParentSize(gnx):
            for pgnx in attrs1[gnx][2]:
                attrs1[pgnx][4] += dsz
                updateParentSize(pgnx)
        updateParentSize(gnx)

        clean_parents(attrs1, nodes1, ns)
    #@+node:vitalije.20180510153738.5: *3* delete_node
    def delete_node(self, pos):
        ( positions, nodes, attrs, levels, expanded, marked) = self.data

        # this node
        i = positions.index(pos)
        gnx = nodes[i]
        sz = attrs[gnx].size
        ns = nodes[i:i+sz]

        if sz > len(positions) - 2:
            # cant delete the only top level node
            return

        # parent node
        pi = parent_index(levels, i)
        pp = positions[pi]
        pgnx = nodes[pi]
        psz = attrs[pgnx].size

        # distance from parent
        pdist = i - pi

        # remove all nodes in subtree
        for pxi in gnx_iter(nodes, pgnx, psz-sz):
            a = pxi + pdist
            b = a + sz
            del nodes[a:b]
            del positions[a:b]
            del levels[a:b]

        # now reduce sizes of all ancestors
        update_size(attrs, pgnx, -sz)

        attrs[gnx].parents.remove(pgnx)
        self._update_children(pgnx)
        clean_parents(attrs, nodes, ns)
    #@+node:vitalije.20180515122209.1: *3* display_items
    def display_items(self, skip=0, count=None):
        (positions, nodes, attrs, levels, expanded, marked) = self.data
        Npos = len(positions)
        if count is None: count = Npos

        selInd = self.selectedIndex
        i = 1
        while count > 0 and i < Npos:
            gnx = nodes[i]
            pos = positions[i]
            h, b, ps, chn, sz = attrs[gnx]
            exp = (pos in expanded)
            if skip > 0:
                skip -= 1
            else:
                count -= 1
                iconVal = 1 if b else 0
                iconVal += 2 if gnx in marked else 0
                iconVal += 4 if len(ps) > 1 else 0
                if chn:
                    pmicon = 'minus' if exp else 'plus'
                else:
                    pmicon = 'none'
                yield pos, gnx, h, levels[i], pmicon, iconVal, selInd == i
            if chn and exp:
                i += 1
            else:
                i += sz
    #@+node:vitalije.20180515155021.1: *3* select_next_node
    def select_next_node(self, ev=None):
        (positions, nodes, attrs, levels, expanded, marked) = self.data
        i = self.selectedIndex
        if i < 0: i = 1
        if self.selectedPosition in expanded:
            i += 1
        else:
            i += attrs[nodes[i]].size
        if i < len(positions):
            self.selectedPosition = positions[i]
            self._selectedIndex = i
            return nodes[i]

    #@+node:vitalije.20180515155026.1: *3* select_prev_node
    def select_prev_node(self, ev=None):
        (positions, nodes, attrs, levels, expanded, marked) = self.data
        i = self.selectedIndex
        if i < 0: i = 1
        j = j0 = 1
        while j < i:
            j0 = j
            if positions[j] in expanded:
                j += 1
            else:
                j += attrs[nodes[j]].size
        self.selectedPosition = positions[j0]
        self._selectedIndex = j0
        return nodes[j0]
    #@+node:vitalije.20180516103325.1: *3* select_node_left
    def select_node_left(self, ev=None):
        '''If currently selected node is collapsed or has no
           children selects parent node. If it is expanded and
           has children collapses selected node'''
        (positions, nodes, attrs, levels, expanded, marked) = self.data
        if self.selectedPosition in expanded:
            expanded.remove(self.selectedPosition)
            return
        i = self.selectedIndex
        if i < 2:return
        pi = up_level_index(levels, i)
        if pi == 0:
            # this is top level node
            # let's find previous top level (level=1) node.
            pi = levels.rfind(b'\x01', 0, i - 1)

        self.selectedPosition = positions[pi]
        self._selectedIndex = pi
        return nodes[pi]
    #@+node:vitalije.20180516105003.1: *3* select_node_right
    def select_node_right(self, ev=None):
        '''If currently selected node is collapsed, expands it.
           In any case selects next node.'''
        (positions, nodes, attrs, levels, expanded, marked) = self.data
        i = self.selectedIndex
        if -1 < i < len(nodes) - 1:
            hasChildren = levels[i] < levels[i + 1]
            p = self.selectedPosition
            if hasChildren and p not in expanded:
                expanded.add(p)
            self.selectedPosition = positions[i + 1]
            self._selectedIndex = i + 1
            return nodes[i + 1]
    #@+node:vitalije.20180518124047.1: *3* visible_positions
    @property
    def visible_positions(self):
        if self._visible_positions_serial != self._visible_positions_last:
            return self.refresh_visible_positions()
        return self._visible_positions

    def invalidate_visual(self):
        self._visible_positions_serial += 1

    def refresh_visible_positions(self):
        self._visible_positions_last = self._visible_positions_serial
        (positions, nodes, attrs, levels, expanded, marked) = self.data
        def it():
            j = 1
            N = len(positions)
            while j < N:
                p1 = positions[j]
                yield p1
                if p1 in expanded:
                    j += 1
                else:
                    j += attrs[nodes[j]].size
        self._visible_positions = tuple(it())
        return self._visible_positions
    #@+node:vitalije.20180516150626.1: *3* subtree
    def subtree(self, pos):
        (positions, nodes, attrs, levels, expanded, marked) = self.data
        i = positions.index(pos)
        gnx = nodes[i]
        sz = attrs[gnx].size
        t = LeoTreeModel()
        (tpositions, tnodes, tattrs, tlevels, texpanded, tmarked) = t.data
        tpositions.extend(positions[i:i+sz])
        lev0 = levels[i]
        tlevels.extend((x - lev0) for x in levels[i:i+sz])
        tnodes.extend(nodes[i:i+sz])

        texpanded.update(expanded & set(tpositions))
        knownGnx = set(tnodes)
        tmarked.update(knownGnx & marked)

        for x in tnodes:
            h, b, ps, chn, sz = attrs[x]
            ps = [y for y in ps if y in knownGnx]
            tattrs[x] = NData(h, b, ps, chn[:], sz)

        return t

    #@+node:vitalije.20180516132431.1: *3* promote
    def promote(self, pos):
        '''Makes following siblings of pos, children of pos'''
        (positions, nodes, attrs, levels, expanded, marked) = self.data
        # node at position pos
        i = positions.index(pos)
        gnx = nodes[i]
        lev0 = levels[i]

        # parent node
        pi = up_level_index(levels, i)
        pp = positions[pi]
        pgnx = nodes[pi]
        psz = attrs[pgnx].size

        # index of node after tree
        after = pi + psz

        # remember originial size
        oldsize = attrs[gnx].size
        A = i + oldsize

        # remember distance from parent
        pdist = i - pi

        # check danger clones
        if gnx in nodes[A:after]:
            #print('warning node can not be in its own subtree')
            return

        # adjust size of this node
        attrs[gnx].size = oldsize + after - A
        #@+others
        #@+node:vitalije.20180517160150.1: *4* 1. promote this part of outline
        chindex = levels[pi+1:i].count(levels[pi] + 1)
        j = A # iterator of following siblings
        while j < after:
            cgnx = nodes[j]
            h, b, ps, chn, sz = attrs[cgnx]

            # let's replace pgnx with gnx in parents
            ps[ps.index(pgnx)] = gnx

            # append to children of this node
            attrs[gnx].children.append(cgnx)

            del attrs[pgnx].children[chindex+1]

            # next sibling
            j += sz

        def inc_levels(m, n):
            levels[m:n] = [x + 1 for x in levels[m:n]]

        done_positions = []
        for pxi in gnx_iter(nodes, pgnx, psz):
            inc_levels(pxi + pdist + oldsize, pxi + psz)
            done_positions.append(positions[pxi + pdist])

        #@+node:vitalije.20180517160237.1: *4* 2. update clones of this node in outline
        # we have already shifted right (increased level) for
        # all existing following siblings in outline. There was no
        # real insertion in positions, nodes and levels arrays
        #
        # Now we need to visit all other clones of this node
        # and insert nodes corresponding to following siblings

        # prepare data for insertion
        sibgnx = nodes[A:after]
        siblev = [x-lev0  for x in levels[A:after]]

        for ii in gnx_iter(nodes, gnx, oldsize):
            if positions[ii] in done_positions:
                # we have already done our job here
                continue

            # old index of after tree
            jj = ii + oldsize

            # insert in nodes
            nodes[jj:jj] = sibgnx

            # insert levels adjusted by level of this clone
            lev1 = levels[ii]
            levels[jj:jj] = [x + lev1 for x in siblev]

            # we need new positions for inserted nodes
            npos = [random.random() for x in siblev]
            positions[jj:jj] = npos

        #@+node:vitalije.20180517160615.1: *4* 3. update sizes in outline
        allparents = attrs[gnx][2][:]
        allparents.remove(pgnx) # for one of them there is no need to change size
        for x in allparents:
            update_size(attrs, x, after-A)
        #@-others

        # finally to show indented nodes
        expanded.add(pos)

    #@+node:vitalije.20180617204800.1: *3* promote_children
    def promote_children(self, pos):
        '''Turns children to siblings of pos'''
        (positions, nodes, attrs, levels, expanded, marked) = self.data

        # this node
        i = positions.index(pos)
        gnx = nodes[i]
        h, b, mps, chn, sz0 = attrs[gnx]
        lev0 = levels[i]

        if sz0 == 1:
            # there are no children to promote
            return

        # parent node
        pi = up_level_index(levels, i)
        pp = positions[pi]
        pgnx = nodes[pi]
        pchn, psz = attrs[pgnx][3:5]
        pdist = i - pi
        chindex = levels[pi:i].count(lev0)

        # relink child nodes
        attrs[pgnx].children[chindex+1:chindex+1] = attrs[gnx].children
        for x in attrs[gnx].children:
            psx = attrs[x].parents
            psx[psx.index(gnx)] = pgnx
        del attrs[gnx].children[:]

        #@+others
        #@+node:vitalije.20180617210617.1: *4* do_one_node
        def do_one_node(i):
            levels[i+1:i+sz0] = [x-1 for x in levels[i+1:i+sz0]]
        #@-others

        attrs[gnx].size = 1  # size of this node

        done_positions = []

        # first we process this parent node throughout the outline
        for pxi in gnx_iter(nodes, pgnx, psz):
            do_one_node(pxi+pdist)
            done_positions.append(positions[pxi+pdist])

        # we need to delete node data from every remaining position
        for pxi in gnx_iter(nodes, gnx, 1):
            if positions[pxi] not in done_positions:
                ai, bi = pxi+1, pxi+sz0
                del positions[ai:bi]
                del levels[ai:bi]
                del nodes[ai:bi]

        # all parent nodes except pgnx need to update their size
        for x in mps:
            if x == pgnx:
                pgnx = None # we don't want to skip this twice
                continue
            update_size(attrs, x, 1 - sz0)
    #@+node:vitalije.20180617170844.1: *3* indent_node
    def indent_node(self, pos):
        '''Moves right node at position pos'''
        (positions, nodes, attrs, levels, expanded, marked) = self.data

        # this node
        i = positions.index(pos)
        if levels[i-1] == levels[i] - 1:
            # if there is no previous siblings node
            # can't be moved right
            return
        gnx = nodes[i]
        h, b, mps, chn, sz0 = attrs[gnx]
        lev0 = levels[i]

        # parent node
        pi = up_level_index(levels, i)
        pp = positions[i]
        pgnx = nodes[pi]
        psz = attrs[pgnx].size

        chindex = levels[pi:i].count(lev0)
        pdist = i - pi

        # new parent
        npi = prev_sibling_index(levels, pi, i)
        npp = positions[npi]
        npgnx = nodes[npi]

        if npgnx in nodes[i:i+sz0]:
            # can not move node in its own subtree
            return

        # link node to the new parent
        mps[mps.index(pgnx)] = npgnx
        attrs[npgnx].children.append(gnx)

        del attrs[pgnx].children[chindex]

        # indent nodes in all clones of parent node
        done_positions = []
        for pxi in gnx_iter(nodes, pgnx, psz):
            a = pxi + pdist
            b = a + sz0
            done_positions.append(positions[a])
            done_positions.append(positions[pxi+npi-pi])
            levels[a:b] = (x+1 for x in levels[a:b])

        # preserve nodes that are being indented
        ns = nodes[i:i+sz0]
        levs = levels[i:i+sz0]

        # now we need to insert it to the destination parent and update outline
        sz1 = attrs[npgnx].size
        for pxi in gnx_iter(nodes, npgnx, sz1):
            if positions[pxi] not in done_positions:
                a = pxi + sz1
                positions[a:a] = [random.random() for x in levs]
                dl = levels[pxi] + 1 - levs[0]
                levels[a:a] = [x+dl for x in levs]
                nodes[a:a] = ns

        update_size(attrs, pgnx, -sz0)
        update_size(attrs, npgnx, sz0)
    #@+node:vitalije.20180517183334.1: *3* dedent_node
    def dedent_node(self, pos):
        '''Moves node left'''
        (positions, nodes, attrs, levels, expanded, marked) = self.data

        # this node
        i = positions.index(pos)
        if levels[i] == 1:
            # can't move left
            return

        gnx = nodes[i]

        # parent node
        pi = up_level_index(levels, i)
        pp = positions[pi]
        pgnx = nodes[pi]
        psz = attrs[pgnx].size

        h, b, mps, chn, sz0 = attrs[gnx]

        # grandparent node
        gpi = up_level_index(levels, pi)
        gpp = positions[pi]
        gpgnx = nodes[gpi]

        di0 = i - gpi
        di1 = di0 + sz0
        di2 = pi - gpi
        di3 = di2 + psz


        def movedata(j, ar):
            ar[j+di0: j+di3] = ar[j+di1:j+di3] + ar[j+di0:j+di1]

        def move_levels(j):
            a = j + di0
            b = j + di1
            levels[a:b] = [x-1 for x in levels[a:b]]
            movedata(j, levels)

        donepos = []
        for gxi in gnx_iter(nodes, gpgnx, attrs[gpgnx].size):
            donepos.append(positions[gxi + di2])
            movedata(gxi, positions)
            movedata(gxi, nodes)
            move_levels(gxi)

        for pxi in gnx_iter(nodes, pgnx, psz-sz0):
            if positions[pxi] not in donepos:
                a = pxi + di0 - di2
                b = a + sz0
                del positions[a:b]
                del nodes[a:b]
                del levels[a:b]

        update_size(attrs, pgnx, -sz0)
        update_size(attrs, gpgnx, sz0)

        # replace parent with grandparent
        mps[mps.index(pgnx)] = gpgnx

        self._update_children(pgnx)
        self._update_children(gpgnx)
    #@+node:vitalije.20180518062711.1: *3* prev_visible_index
    def prev_visible_index(self, pos):
        '''Assuming this node is visible, search for previous
           visible node.'''
        (positions, nodes, attrs, levels, expanded, marked) = self.data

        # this node
        i = positions.index(pos)

        # parent node
        pi = up_level_index(levels, i)

        j = pi + 1
        A = pi
        while j < i:
            A = j
            if positions[j] in expanded:
                j += 1
            else:
                j += attrs[nodes[j]].size
        return A
    #@+node:vitalije.20180518082938.1: *3* next_visible_index
    def next_visible_index(self, pos):
        '''Assuming this node is visible, search for previous
           visible node.'''
        (positions, nodes, attrs, levels, expanded, marked) = self.data

        # this node
        i = positions.index(pos)

        if pos in expanded:
            return i + 1
        return i + attrs[nodes[i]].size
    #@+node:vitalije.20180518055719.1: *3* move_node_up
    def move_node_up(self, pos):
        '''Moves node one step towards the top of outline'''
        (positions, nodes, attrs, levels, expanded, marked) = self.data
        if pos == positions[1]:
            # already at the top
            return
        i = positions.index(pos)
        B = self.prev_visible_index(pos)

        return self.move_node_to_index(i, B, up_level_index(levels, B))
    #@+node:vitalije.20180518055819.1: *3* move_node_down
    def move_node_down(self, pos):
        '''Moves node one step towards the end of outline'''
        (positions, nodes, attrs, levels, expanded, marked) = self.data

        i = positions.index(pos)
        gnx = nodes[i]
        sz = attrs[gnx].size
        j = i + sz

        if j  == len(positions):
            # already at the end
            return

        sz1 = attrs[nodes[j]].size
        if sz1 > 1 and positions[j] in expanded:
            dest = j
            j += 1
        else:
            dest = up_level_index(levels, j)
            j += sz1

        return self.move_node_to_index(i, j, dest)
    #@+node:vitalije.20180617165706.1: *3* algorithm
    #@+at
    # Whenever we move node in tree the following should be kept in mind:
    # 
    # if the destination parent has a single location within the outline then
    # all that needs to be done is to transfer this node data to the destination
    # parent and update all other instances of this parent node
    # 
    # if current parent has only one occurrence within the outline then it is
    # safe to cut out this node and then insert it to the destination node
    # and update all instances of destination parent.
    # 
    # 1. mk copy of this node data
    # 2. delete it from all occurrences of the parent node
    # 3. for each position px of destination parent
    #         if px is required destination position
    #             paste node data with original positions
    #         else
    #             paste node data with new positions
    #@+node:vitalije.20180518071115.1: *3* move_node_to_index
    def move_node_to_index(self, A, B, npi):
        '''Moves node from index A to index B'''
        (positions, nodes, attrs, levels, expanded, marked) = self.data
        if B == A + 1:
            return
        gnx = nodes[A]
        h, b, mps, chn, sz0 = attrs[gnx]

        pi = up_level_index(levels, A)
        pp = positions[pi]
        pgnx = nodes[pi]
        chindexA = levels[pi:A].count(levels[A])

        #npi = up_level_index(levels, B)
        np = positions[npi]
        ndp = positions[B] if B < len(positions) else 2
        npgnx = nodes[npi]
        chindexB = levels[npi:B].count(levels[npi]+1)

        # check for self linking
        if npgnx in nodes[A:A+sz0]:
            #print('Warning can not move node to its own subtree')
            return
        if npgnx == pgnx and chindexA == chindexB:
            # moving from one clone to another at the same index
            # nothing to do
            return

        # relink
        if pgnx != npgnx:
            mps[mps.index(pgnx)] = npgnx

            del attrs[pgnx].children[chindexA]
            attrs[npgnx].children.insert(chindexB, gnx)
        else:
            xs = attrs[pgnx].children
            if chindexB > chindexA:
                xs.insert(chindexB, gnx)
                del xs[chindexA]
            else:
                del xs[chindexA]
                xs.insert(chindexB, gnx)

        # store node data
        ps = positions[A:A+sz0]
        ns = nodes[A:A+sz0]
        lev0 = levels[A]
        levs = [x-lev0 for x in levels[A:A+sz0]]

        # delete from current parent
        delete_from_all_parent_instances(self.data, pgnx, A - pi, A - pi + sz0)
        update_size(attrs, pgnx, -sz0)

        # most likely index was changed by deletion, lets recalculate
        npi = positions.index(np)
        if ndp < 1:
            npdist = positions.index(ndp) - npi
        else: 
            npdist = len(positions) - npi

        insert_node_data(self.data, (ps, ns, levs), npdist + npi, npi)
        npsz = attrs[npgnx].size
        for pxi in gnx_iter(nodes, npgnx, npsz):
            if positions[pxi] != np:
                ps = [random.random() for x in levs]
                insert_node_data(self.data, (ps, ns, levs), npdist + pxi, pxi)
        update_size(attrs, npgnx, sz0)
    #@+node:vitalije.20180613193129.1: *3* collapse_all
    def collapse_all(self):
        self.data.expanded.clear()
    #@+node:vitalije.20180630192650.1: *3* ensure_visible
    def ensure_visible(self, pos):
        (positions, nodes, attrs, levels, expanded, marked) = self.data
        i = positions.index(pos)
        make_visible(self.data, i)

    def ensure_visible_index(self, i):
        make_visible(self.data, i)
    #@+node:vitalije.20180613193239.1: *3* expand_all
    def expand_all(self):
        (positions, nodes, attrs, levels, expanded, marked) = self.data
        expanded.update(x for i,x in enumerate(positions) if attrs[nodes[i]].size > 1)
    #@+node:vitalije.20180613193259.1: *3* toggle
    def toggle(self, p):
        if p in self.data.expanded:
            self.data.expanded.remove(p)
        else:
            self.data.expanded.add(p)
    #@+node:vitalije.20180518155629.1: *3* body_change
    def body_change(self, newbody):
        i = self.selectedIndex
        if i < 0:
            return
        (positions, nodes, attrs, levels, expanded, marked) = self.data

        gnx = nodes[i]
        a = attrs[gnx]
        a.b = newbody
    #@+node:vitalije.20180518155645.1: *3* p_to_string
    def p_to_string(self, p, delim_st, delim_en):
        '''Produces and returns string content of external file
           that corresponds to the given position, using provided
           delimiters.'''
        return ''.join(p_to_chunks(self.data, p, delim_st, delim_en))
    #@+node:vitalije.20180529164742.1: *3* p_to_autostring
    def p_to_autostring(self, p):
        '''Produces and returns string content of external auto file
           that corresponds to the given position.'''

        return ''.join(x[0] for x in p_to_autolines(self.data, p))

    #@+node:vitalije.20180518155655.1: *3* to_bytes
    def to_bytes(self):
        '''Returns pickled data of this model'''
        return pickle.dumps(self.data)
    #@+node:vitalije.20180518155712.1: *3* static
    #@+node:vitalije.20180614214938.1: *4* from_bytes
    @staticmethod
    def from_bytes(bs):
        ltm = LeoTreeModel()
        return ltm.restore_from_bytes(bs)


    #@+node:vitalije.20180614215006.1: *4* from_xml
    @staticmethod
    def from_xml(fname):
        return load_leo(fname)
    #@+node:vitalije.20180614215013.1: *4* load_full
    @staticmethod
    def load_full(fname):
        return load_leo_full(fname)
    #@+node:vitalije.20180614215025.1: *3* load_external_files
    def load_external_files(self, loaddir):
        return load_external_files(self, loaddir)
    #@+node:vitalije.20180518155716.1: *3* restore_from_bytes
    def restore_from_bytes(self, bs):
        data = pickle.loads(bs)
        self.data = LTMData(*data)
        self.invalidate_visual()
        return self
    #@+node:vitalije.20180614162047.1: *3* pre_cmd
    def pre_cmd(self, extra=None):
        i = self._undopos + 1
        x = copy_ltmdata(self.data), extra
        self._undostack[i:] = [x]
        self._undopos = i


    #@+node:vitalije.20180614214753.1: *3* undo
    def undo(self):
        i = self._undopos
        if i < 0:
            return
        self.data, extra = self._undostack[i]
        self._undopos = i - 1
        return extra
    #@+node:vitalije.20180614214757.1: *3* redo
    def redo(self):
        i = self._undopos + 1
        if i > len(self._undostack):
            return
        self._undopos = i
        self.data, extra = self._undostack[i]
        return extra
    #@-others


#@+node:vitalije.20180617172952.1: ** low level utils
#@+at
# This utility functions are low level. They don't preserve validity of tree data.
# Mostly they manipulate raw data from model's lists and dictionaries
#@+node:vitalije.20180630192218.1: *3* make_visible
def make_visible(ltmdata, i):
    (positions, nodes, attrs, levels, expanded, marked) = ltmdata
    while i > 1:
        pi = up_level_index(levels, i)
        p = positions[pi]
        expanded.add(p)
        i = pi
#@+node:vitalije.20180617193158.1: *3* insert_node_data
def insert_node_data(ltmdata, insertdata, i, pi):
    (positions, nodes, attrs, levels, expanded, marked) = ltmdata
    ps, ns, levs = insertdata

    positions[i:i] = ps
    dl = levels[pi] - levs[0] + 1
    levels[i:i] = [x + dl for x in levs]
    nodes[i:i] = ns
#@+node:vitalije.20180617172926.1: *3* delete_from_all_parent_instances
def delete_from_all_parent_instances(ltmdata, pgnx, a, b):
    (positions, nodes, attrs, levels, expanded, marked) = ltmdata
    for pxi in gnx_iter(nodes, pgnx, attrs[pgnx].size+a-b):
        ai, bi = pxi+a, pxi+b
        del positions[ai:bi]
        del levels[ai:bi]
        del nodes[ai:bi]
#@+node:vitalije.20180617172650.1: *3* delete_from_parent_instance
def delete_from_parent_instance(ltmdata, j, a, b):
    (positions, nodes, attrs, levels, expanded, marked) = ltmdata
    ai = j + a
    bi = j + b
    del positions[ai:bi]
    del levels[ai:bi]
    del nodes[ai:bi]
#@+node:vitalije.20180617173507.1: *3* update_size
def update_size(attrs, gnx, delta):
    def ds(x):
        attrs[x].size += delta
        for x1 in attrs[x].parents:
            ds(x1)
    ds(gnx)
#@+node:vitalije.20180618132111.1: *3* clean_parents
def clean_parents(attrs, nodes, ns):
    missing = set(ns) - set(nodes)
    for x in ns:
        ps = attrs[x].parents
        ps[:] = [x1 for x1 in ps if x1 not in missing]
    missing = set(attrs) - set(nodes)
    for x in missing:
        attrs.pop(x)
#@+node:vitalije.20180617174721.1: *3* prev_sibling_index
def prev_sibling_index(levels, pi, i):
    return levels.rfind(levels[i], pi, i)
#@+node:vitalije.20180510103732.1: ** load_derived_file
def load_derived_file(lines):
    '''Returns generator which yields tuples:
       (gnx, h, b, level)
    '''
    flines = tuple(enumerate(lines))
    first_lines = []
    #@+others
    #@+node:vitalije.20180510155832.1: *3* bunch
    class bunch:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    #@+node:vitalije.20180510103732.2: *3* 1. first_lines & header
    header_pattern = re.compile(r'''
        ^(.+)@\+leo
        (-ver=(\d+))?
        (-thin)?
        (-encoding=(.*)(\.))?
        (.*)$''', re.VERBOSE)

    for i, line in flines:
        m = header_pattern.match(line)
        if m:
            break
        first_lines.append(line)
    else:
        raise ValueError('wrong format, not derived file')
    # m.groups example ('#', '-ver=5', '5', '-thin', None, None, None, '')
    delim_st = m.group(1)
    delim_en = m.group(8)

    leo_marker = delim_st + '@'
    #@+node:vitalije.20180510103732.3: *3* 2. patterns
    #@+others
    #@+node:vitalije.20180510103732.4: *4* get_patterns
    # this function can be avoided and its calculations made inline
    # but if we should support change in delims during the read
    # process it would be better if we have this as a function
    def get_patterns(delim_st, delim_en):
        if delim_en:
            dlms = re.escape(delim_st), re.escape(delim_en)
            ns_src = r'^(\s*)%s@\+node:([^:]+): \*(\d+)?(\*?) (.*?)%s$'%dlms
            sec_src = r'^(\s*)%s@(\+|-)<{2}[^>]+>>%s$'%dlms
            oth_src = r'^(\s*)%s@(\+|-)others%s\s*$'%dlms
            all_src = r'^(\s*)%s@(\+|-)all%s\s*$'%dlms
            code_src = r'^%s@@c(ode)?%s$'%dlms
            doc_src = r'^%s@\+(at|doc)?(\s.*?)?%s$'%dlms
        else:
            dlms = re.escape(delim_st)
            ns_src = r'^(\s*)%s@\+node:([^:]+): \*(\d+)?(\*?) (.*)$'%dlms
            sec_src = r'^(\s*)%s@(\+|-)<{2}.+?>>$'%dlms
            oth_src = r'^(\s*)%s@(\+|-)others\s*$'%dlms
            all_src = r'^(\s*)%s@(\+|-)all\s*$'%dlms
            code_src = r'^%s@@c(ode)?$'%dlms
            doc_src = r'^%s@\+(at|doc)?(\s.*?)?'%dlms + '\n'
        return bunch(
            node_start = re.compile(ns_src),
            section    = re.compile(sec_src),
            others     = re.compile(oth_src, re.DOTALL),
            all        = re.compile(all_src, re.DOTALL),
            code       = re.compile(code_src),
            doc        = re.compile(doc_src),
        )

    #@-others
    patterns = get_patterns(delim_st, delim_en)

    #@+node:vitalije.20180510103732.5: *3* 3. top node
    #@+at
    #    Find beginning of top (root node) of this derived file.
    #    We expect zero or more first lines before leo header line.
    # 
    #    Leo header line will give usefull information such as delimiters.
    # 
    #@+node:vitalije.20180510103732.6: *4* nodes bunch
    # this is where we will collect all the data from input
    nodes = bunch(
        # level must contain lists of node levels in order they appear in input
        # this is to support at-all directive which will write clones several times.
        level = defaultdict(list),

        # contains headline for each node
        head = {},

        # contains lines of body text for each node
        body = defaultdict(list),

        # this is list which will store the order of nodes in derived file
        # that is the order in which we will dump nodes once we have consumed
        # all input lines
        gnxes = [],
    )

    #@+node:vitalije.20180510103732.7: *4* set_node
    #@+at utility function to set data from regex match object from sentinel line
    #   see node_start pattern. groups[1 - 5] are:
    #   (indent, gnx, level-number, second star, headline)
    #       1      2         3            4          5
    #   returns gnx
    #@@c
    def set_node(m):
        gnx = m.group(2)
        lev = int(m.group(3)) if m.group(3) else 1 + len(m.group(4))
        nodes.level[gnx].append(lev)
        nodes.head[gnx] = m.group(5)
        nodes.gnxes.append(gnx)
        return gnx

    #@+node:vitalije.20180510103732.8: *4* doc_skip
    #@+at if we are using delim_en for closing comment, 
    #   then doc parts start with delim_st alone on line
    #   and doc parts end with delim_en alone on line
    # 
    #    so whenever we encounter any of this lines, 
    #    we have just to skip and nothing should be added to body.
    #@@c
    doc_skip = (delim_st + '\n', 
                delim_en + '\n')
    #@+node:vitalije.20180510103732.9: *4* start top node
    topnodeline = flines[len(first_lines) + 1][1] # line after header line

    m = patterns.node_start.match(topnodeline)
    topgnx = set_node(m)

    # append first lines if we have some
    nodes.body[topgnx] = ['@first '+ x for x in first_lines]
    assert topgnx, 'top node line [%s] %d first lines'%(topnodeline, len(first_lines))

    # this will keep track of current gnx and indent whenever we encounter 
    # at+others or at+<section> or at+all
    stack = []

    in_all = False
    in_doc = False


    # spelling of at-verbatim sentinel
    verbline = delim_st + '@verbatim' + delim_en + '\n'
    arline = delim_st + '@afterref' + delim_en + '\n'

    verbatim = False # keeps track whether next line is to be processed or not
    afterref = False # keeps track of after ref line
    #@+node:vitalije.20180510103732.10: *3* 4. iterate lines
    # we need to skip twice the number of first_lines, one header line
    # and one top node line
    start = 2 * len(first_lines) + 2

    # keeps track of current indentation
    indent = 0 

    # keeps track of current node that we are reading
    gnx = topgnx

    # list of lines for current node
    body = nodes.body[gnx]


    for i, line in flines[start:]:
        # child nodes may if necessary shortcut this loop
        # using continue or let the line fall through to 
        # the end of loop
        if line.strip().startswith(leo_marker):
            #@+others
            #@+node:vitalije.20180510103732.11: *4* handle verbatim
            if verbatim:
                # previous line was verbatim sentinel. let's append this line as it is
                if afterref:
                    body[-1] = body[-1][:-1] + line
                    afterref = False
                else:
                    body.append(line)
                verbatim = False # (next line should be normally processed)
                continue

            if line == verbline:
                # this line is verbatim sentinel, next line should be appended as it is
                verbatim = True
                continue

            _ari = line.find(arline)
            if _ari > -1 and line[:_ari].isspace():
                verbatim = True
                afterref = True
                continue
            #@+node:vitalije.20180510103732.12: *4* handle indent
            # is indent still valid?
            if indent and line[:indent].isspace() and len(line) > indent:
                # yes? let's strip unnecessary indentation
                line = line[indent:]

            #@+node:vitalije.20180510103732.13: *4* handle at_all
            m = patterns.all.match(line)
            if m:
                in_all = m.group(2) == '+' # is it opening or closing sentinel
                if in_all:
                    # opening sentinel
                    body.append('@all\n')
                    # keep track which node should we continue to build
                    # once we encounter closing at-all sentinel
                    stack.append((gnx, indent))
                else:
                    # this is closing sentinel
                    # let's restore node where we started at-all directive
                    gnx, indent = stack.pop()
                    # restore body which should receive next lines
                    body = nodes.body[gnx]
                continue
            #@+node:vitalije.20180510103732.14: *4* handle at_others
            m = patterns.others.match(line)
            if m:
                in_doc = False
                if m.group(2) == '+': # is it opening or closing sentinel
                    # opening sentinel
                    body.append(m.group(1) + '@others\n')
                    # keep track which node should we continue to build
                    # once we encounter closing at-others sentinel
                    stack.append((gnx, indent))
                    indent += m.end(1) # adjust current identation
                else:
                    # this is closing sentinel
                    # let's restore node where we started at-others directive
                    gnx, indent = stack.pop()
                    # restore body which should receive next lines
                    body = nodes.body[gnx]
                continue

            #@+node:vitalije.20180510103732.15: *4* handle doc
            if not in_doc:
                # we are not yet in doc part
                # maybe we are at the beginning of doc part
                m = patterns.doc.match(line)
                if m:
                    # yes we are at the beginning of doc part
                    # was it @+at or @+doc?
                    doc = '@doc' if m.group(1) == 'doc' else '@'
                    doc2 = m.group(2) or '' # is there any text on first line?
                    if doc2:
                        # start doc part with some text on the same line
                        body.append('%s%s\n'%(doc, doc2))
                    else:
                        # no it is only directive on this line
                        body.append(doc + '\n')

                    # following lines are part of doc block
                    in_doc = True
                    continue
            #@+node:vitalije.20180510103732.16: *4* handle_code
            if in_doc:
                # we are in doc part

                # when using both delimiters, doc block starts with first delimiter
                # alone on line and at the end of doc block end delimiter is also
                # alone on line. Both of this lines should be skipped

                if line in doc_skip: continue

                # maybe this line ends doc part and starts code part?
                m = patterns.code.match(line)
                if m:
                    # yes, this line is at-c or at-code line

                    in_doc = False  # stop building doc part

                    # append directive line
                    body.append('@code\n' if m.group(1) else '@c\n')
                    continue
            #@+node:vitalije.20180510103732.17: *4* handle section ref
            m = patterns.section.match(line)
            if m:
                in_doc = False

                if m.group(2) == '+': # is it opening or closing sentinel
                    # opening sentinel
                    ii = m.end(2) # before <<
                    body.append(m.group(1) + line[ii:])

                    # keep track which node should we continue to build
                    # once we encounter closing at-<< sentinel
                    stack.append((gnx, indent))

                    indent += m.end(1) # adjust current identation
                else:
                    # this is closing sentinel

                    # let's restore node where we started at+<< directive
                    gnx, indent = stack.pop()

                    # restore body which should receive next lines
                    body = nodes.body[gnx]
                continue
            #@+node:vitalije.20180510103732.18: *4* handle node_start
            m = patterns.node_start.match(line)
            if m:
                in_doc = False
                gnx = set_node(m)
                if len(nodes.level[gnx]) > 1:
                    # clone in at-all
                    # let it collect lines in throwaway list
                    body = []
                else:
                    body = nodes.body[gnx]
                continue

            #@+node:vitalije.20180510103732.19: *4* handle @-leo
            if line.startswith(delim_st + '@-leo'):
                break
            #@+node:vitalije.20180510103732.20: *4* handle directive
            if line.startswith(delim_st + '@@'):
                ii = len(delim_st) + 1 # on second '@'

                # strip delim_en if it is set or just '\n'
                jj = line.rfind(delim_en) if delim_en else -1

                # append directive line
                body.append(line[ii:jj] + '\n')
                continue

            #@+node:vitalije.20180510103732.21: *4* handle in_doc
            if in_doc and not delim_en:
                # when using just one delimiter (start)
                # doc lines start with delimiter + ' '
                body.append(line[len(delim_st)+1:])
                continue
            # when delim_en is not '', doc part starts with one start delimiter and \n
            # and ends with end delimiter followed by \n
            # in that case doc lines are unchcanged
            #@-others


        # nothing special about this line, let's append it to current body
        body.append(line)


    if i + 1 < len(flines):
        nodes.body[topgnx].extend('@last %s'%x for x in flines[i+1:])

    #@+node:vitalije.20180510103732.22: *3* 5. dump nodes
    for gnx in nodes.gnxes:
        b = ''.join(nodes.body[gnx])
        h = nodes.head[gnx]
        lev = nodes.level[gnx].pop(0)
        yield gnx, h, b, lev-1

    #@-others
#@+node:vitalije.20180510191554.1: ** ltm_from_derived_file
def ltm_from_derived_file(fname):
    '''Reads external file and returns tree model.'''
    with open(fname, 'rt') as inp:
        lines = inp.read().splitlines(True)
        parents = defaultdict(list)
        def viter():
            stack = [None for i in range(256)]
            lev0 = 0
            seen = set()
            for gnx, h, b, lev in load_derived_file(lines):
                ps = parents[gnx]
                cn = []
                s = [1]
                stack[lev] = [gnx, h, b, lev, s, ps, cn, gnx not in seen]
                seen.add(gnx)
                if lev:
                    # add parent gnx to list of parents
                    if stack[lev-1][7]:
                        ps.append(stack[lev - 1][0])
                    if lev > lev0:
                        # parent level is lev0
                        # add this gnx to list of children in parent
                        stack[lev0][6].append(gnx)
                    else:
                        # parent level is one above 
                        # add this gnx to list of children in parent
                        stack[lev - 1][6].append(gnx)
                lev0 = lev

                # increase size of every node in current stack
                for x in stack[:lev]:
                    x[4][0] += 1

                # finally yield this node
                yield stack[lev][:-1]

        nodes = tuple(viter())
        return nodes2treemodel(nodes)

#@+node:vitalije.20180512100218.1: ** chunks2lines
def chunks2lines(it):
    '''Modifies iterator that yields arbitrary chunks of text
       to iterator that yields complete lines of text. This is
       used in testing for comparison with lines in external
       files.'''
    buf = ''.join(it)
    for line in buf.splitlines(True):
        yield line
#@+node:vitalije.20180512100206.1: ** p_to_lines
def p_to_lines(ltmdata, pos, delim_st, delim_en):
    '''Returns iterator of lines representing the derived
       file (format version: 5-thin) of given position
       using provided delimiters.'''
    it = p_to_chunks(ltmdata, pos, delim_st, delim_en)
    return chunks2lines(it)
#@+node:vitalije.20180511214549.1: ** p_to_chunks
def p_to_chunks(ltmdata, pos, delim_st, delim_en):
    '''Returns iterator of chunks representing the derived
       file (format version: 5-thin) of given position
       using provided delimiters.'''

    for p, line, lln in p_at_file_iterator(ltmdata, pos, delim_st, delim_en):
        yield line
#@+node:vitalije.20180529171633.1: *3* p_at_file_iterator
def p_at_file_iterator(ltmdata, pos, delim_st, delim_en):
    (positions, nodes, attrs, levels, expanded, marked) = ltmdata
    last = []
    pindex = positions.index(pos)
    #@+others
    #@+node:vitalije.20180511214738.1: *4* bunch
    class bunch:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    #@+node:vitalije.20180511214549.2: *4* conf
    conf = bunch(
        delim_st=delim_st,
        delim_en=delim_en,
        in_doc=False,
        in_all=False,
        zero_level=levels[pindex])
    #@+node:vitalije.20180511214549.3: *4* write patterns
    section_pat = re.compile(r'^(\s*)(<{2}.+?>>)(.*)$')

    others_pat = re.compile(r'^(\s*)@others\b', re.M)
        # !important re.M used also in others_iterator

    doc_pattern = re.compile('^(@doc|@)(?:\\s(.*?)\n|\n)$')

    code_pattern = re.compile('^(@code|@c)$')


    # TODO: check if there are more directives that
    #       should be in this pattern
    atdir_pat = re.compile('^@('
        'beautify|'
        'color|'
        'encoding|'
        'killbeautify|'
        'killcolor|'
        'language|'
        'last|'
        'nobeautify|'
        'nocolor-node|'
        'nocolor|'
        'nosearch|'
        'pagewidth|'
        'path|'
        'root|'
        'tabwidth|'
        'wrap)')
    #@+node:vitalije.20180511214549.4: *4* section_ref
    def section_ref(s):
        m = section_pat.match(s)
        if m:
            return m.groups()
        return None, None, None

    #@+node:vitalije.20180512071638.1: *4* shouldBeIgnored
    def shouldBeIgnored(h, b):
        return h.startswith('@ignore') or b.startswith('@ignore') or '\n@ignore' in b

    #@+node:vitalije.20180511214549.5: *4* others_iterator
    def others_iterator(pind):
        # index of node after subtree is pind + <subtree-size>
        after = pind + attrs[nodes[pind]].size

        p1 = pind + 1
        while p1 != after:
            gnx = nodes[p1]
            h, b, ps, chn, sz = attrs[gnx]
            if shouldBeIgnored(h, b) or section_ref(h)[1]:
                p1 += sz  # skip entire p1-subtree 
            else:
                yield p1
                if others_pat.search(b):
                    # skip entire subtree if p1 contains at-others directive
                    p1 += sz
                else:
                    # next node in outline order
                    p1 += 1
    #@+node:vitalije.20180512160226.1: *4* findReference
    def findReference(pind, ref):
        '''Returns index of node with section definition.'''
        gnx = nodes[pind]
        sz = attrs[gnx].size

        # clean and normalize reference
        ref = ref.lower().replace(' ', '').replace('\t', '')

        for i in range(pind+1, pind+sz): # reference must be in subtree
            ignx = nodes[i]
            h, b, ps, chn, sz = attrs[ignx]
            h = h.lower().replace(' ', '').replace('\t', '').lstrip('.')
            # normalize headline
            if h.startswith(ref):
                # found reference definition node
                return i

    #@+node:vitalije.20180511214549.6: *4* section_replacer
    def section_replacer(it):
        for p, w, final,lln in it:
            if final:
                # pass final lines
                yield p, w, final, lln
                continue
            # does this line contain section reference
            indent, sref, after = section_ref(w)
            if sref and not conf.in_doc:
                if conf.in_all:
                    #CHECK: This perhaps can't happen
                    yield p, w, True, lln
                else:
                    p1 = findReference(p, sref)
                    if not p1:
                        raise LookupError('unresolved section reference: %s'%w)
                    yield p, sent_line('@+', sref, indent=indent), True, lln
                    for p2, w2, final,lln2 in all_lines(p1):
                        w2 = indent + w2 if w2 != '\n' else w2
                        yield p2, w2, final, lln2
                    yield p, sent_line('@-', sref, indent=indent), True, lln
                    if after:
                        w2 = sent_line('@afterref', indent=indent) + after + '\n'
                        yield p, w2, True, lln
                    conf.in_doc = False
            else:
                yield p, w, final, lln
    #@+node:vitalije.20180511214549.7: *4* open_node
    def open_node(p):
        gnx = nodes[p]
        lev = levels[p]
        h = attrs[gnx].h
        stlev = star_level(lev - conf.zero_level)
        return  sent_line('@+node:', gnx, stlev, h)

    #@+node:vitalije.20180511214549.8: *4* body_lines
    def body_lines(p):
        plevel = levels[p]
        first = plevel == conf.zero_level
        if not first:
            yield p, open_node(p), True, 0
        conf.in_doc = False
        b = attrs[nodes[p]].b
        blines = b.splitlines(True)
        if b:
            for i, line in enumerate(blines):
                # child nodes should use continue
                # if they need to skip following nodes
                #@+others
                #@+node:vitalije.20180511214549.9: *5* verbatim
                verbatim = needs_verbatim(line)
                if verbatim:
                    yield p, sent_line('@verbatim') + line, True, i + 1
                    continue

                #@+node:vitalije.20180511214549.10: *5* first lines & leo header
                if first:
                    if line.startswith('@first '):
                        yield p, line[7:], True, i + 1
                        continue
                    else:
                        first = False
                        yield p, sent_line('@+leo-ver=5-thin'), True, i + 1
                        yield p, open_node(p), True, i + 1
                        fstr = sent_line('@@first')
                        for k in range(i):
                            yield p, fstr, True, i + 1

                #@+node:vitalije.20180511214549.11: *5* last lines
                if not conf.in_all:
                    if line.startswith('@last '):
                        last.append((line[6:], i + 1))
                        yield p, sent_line('@@last'), True, i + 1
                        continue
                    elif last:
                        raise ValueError('@last must be last line in body')
                        break

                #@-others
                yield p, line, False, i + 1
            if not line.endswith('\n'):
                yield p, '\n', False, len(blines)
    #@+node:vitalije.20180511214549.12: *4* needs_verbatim
    def needs_verbatim(line):
        return line.lstrip().startswith(conf.delim_st + '@')

    #@+node:vitalije.20180511214549.13: *4* others_replacer
    def others_replacer(it):
        for p, w, final, lln in it:
            if final:
                yield p, w, final, lln
                continue
            m = others_pat.match(w)
            if m and not conf.in_doc:
                if conf.in_all:
                    #CHECK: This perhaps can't happen
                    yield p, w, True, lln
                else:
                    indent = m.group(1)
                    w1 = sent_line('@+others',indent=indent)
                    yield p, w1, True, lln
                    for p1 in others_iterator(p):
                        for p2, w2, final, lln2 in all_lines(p1):
                            w2 = indent + w2 if w2 != '\n' else w2
                            yield p2, w2, final, lln2
                    yield p, sent_line('@-others',indent=indent), True, lln
                    conf.in_doc = False
            else:
                yield p, w, final, lln

    #@+node:vitalije.20180511214549.14: *4* atall_replacer
    def atall_replacer(it):
        for p, w, final, lln in it:
            if final:
                yield p, w, final, lln
                continue
            if w == '@all\n':
                conf.in_all = True
                conf.in_doc = False
                yield p, sent_line('@+all'), True, lln
                gnx = nodes[p]
                sz = attrs[gnx].size
                for p1 in range(p + 1, p + sz):
                    for p2, w2, final, lln2 in all_lines(p1):
                        yield p2, w2, final, lln2
                yield p, sent_line('@-all'), True, lln
                conf.in_all = False
                conf.in_doc = False
            else:
                yield p, w, final, lln

    #@+node:vitalije.20180511214549.16: *4* all_lines
    def all_lines(p):
        it = body_lines(p)
        it = atall_replacer(it)
        it = section_replacer(it)
        it = others_replacer(it)
        it = at_adder(it)
        it = at_docer(it)
        return it
    #@+node:vitalije.20180511214549.17: *4* at_docer
    def at_docer(it):
        for p, w, final, lln in it:
            if final or conf.in_all:
                yield p, w, final, lln
                continue
            #@+others
            #@+node:vitalije.20180511214549.18: *5* at, at-doc
            if not conf.in_doc:
                m = doc_pattern.match(w)
                if m:
                    conf.in_doc = True
                    docdir = '@+at' if m.group(1) == '@' else '@+doc'
                    docline = ' ' + m.group(2) if m.group(2) else ''
                    yield p, sent_line(docdir, docline), True, lln
                    if conf.delim_en:
                        yield p, conf.delim_st + '\n', True, lln
                    continue

            #@+node:vitalije.20180511214549.19: *5* at-c at-code
            if conf.in_doc:
                m = code_pattern.match(w)
                if m:
                    if conf.delim_en:
                        yield p, conf.delim_en  + '\n', True, lln
                    yield p, sent_line('@', m.group(1)), True, lln
                    conf.in_doc = False
                    continue

            #@-others
            if conf.in_doc and not conf.delim_en:
                yield p, sent_line(' ', w[:-1]), True, lln
            else:
                yield p, w, False, lln

    #@+node:vitalije.20180511214549.20: *4* at_adder
    def at_adder(it):
        for p, w, final, lln in it:
            if final:
                yield p, w, final, lln
                continue
            m = atdir_pat.match(w)
            if m and not conf.in_all:
                yield p, sent_line('@', w[:-1]), True, lln
            else:
                yield p, w, False, lln

    #@+node:vitalije.20180511214549.21: *4* star_level
    def star_level(lev):
        if lev < 2:
            return [': * ', ': ** '][lev]
        else:
            return ': *%d* '%(lev + 1)

    #@+node:vitalije.20180511214549.22: *4* sent_line
    def sent_line(s1, s2='', s3='', s4='', indent=''):
        return ''.join((indent,conf.delim_st, s1, s2, s3, s4, conf.delim_en, '\n'))

    #@-others
    for p, line, final, lln in all_lines(pindex):
        yield p, line, lln
    yield pos, sent_line('@-leo'), 0
    for line, lln in last:
        yield pos, line, lln
#@+node:vitalije.20180529143138.1: ** new_gnx
def new_gnx_generator(_id):
    nind = lambda:_id + time.strftime('.%Y%m%d%H%M%S', time.localtime())
    curr = [nind()]
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    def wrapper():
        ts = nind()
        if ts != curr[0]:
            curr[0] = ts
            return ts
        return ts + '.' + ''.join(random.choice(chars) for i in range(6))
    return wrapper
new_gnx = new_gnx_generator('vitalije')
#@+node:vitalije.20180529134759.1: ** auto_py
def auto_py(gnx, fname):
    '''Builds outline from py file'''
    return auto_py_from_string(gnx, read_py_file_content(fname))

def read_py_file_content(fname):
    pat = re.compile(br'^[ \t\v]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)')
    with open(fname, 'rb') as inp:
        bsrc = inp.read()
        lines = bsrc.split(b'\n', 2)
        l1 = lines[0] if len(lines) > 0 else b''
        l2 = lines[1] if len(lines) > 1 else b''
        m = pat.match(l1) or pat.match(l2)
        if m:
            enc = m.group(1).decode('ascii')
            if enc == 'iso-latin-1-unix':
                enc = 'latin-1'
        else:
            enc = 'utf-8'
        try:
            src = bsrc.decode(enc)
        except LookupError:
            return ''
        except UnicodeDecodeError:
            print('file:[%s]'%fname)
            print(repr(lines))
            print('enc[%s]'%enc)
            raise
        if '\t' in src:
            src = src.replace('\t', ' '*4)
        return src
#@+node:vitalije.20180529135809.1: ** auto_py_from_string
Line = namedtuple('Line', 'ln st en ind cl isInd cpos txt')
def auto_py_from_string(rgnx, src):
    '''Builds tree model from source string content src with given root gnx'''

    #
    # inside this function attrs is not same as the attrs field of LTMData
    # keys are gnxes and values are lists of [parents, children, head, body]
    #@+others
    #@+node:vitalije.20180529151834.1: *3* srcLines
    def srcLines(src):
        lines = src.splitlines(True)
        if lines[-1] != '\n':lines.append('\n')
        #@+others
        #@+node:vitalije.20180529151834.2: *4* find
        N = len(src)
        def find(chs, i):
            ii = [src.find(x, i) for x in chs]
            ii.append(N)
            return min([x for x in ii if x > -1])
        #@+node:vitalije.20180529151834.3: *4* q3blocks and comments
        q3s = "'''"
        q3d = '"""'
        q3sd = (q3s, q3d)
        q3blocks = []
        comments = []
        i = 0
        while i < N:
            i1 = find(("'", '"', '#'), i)
            if i1 == N:break
            if src.startswith(q3sd, i1):
                i2 = find(q3sd, i1 + 3)
                q3blocks.append((i1, i2))
                i = i2 + 3
            elif src[i1] in ('"', "'"):
                i2 = i1 + 1
                while src[i2] != src[i1]:
                    if src[i2] == '\\':
                        i2 += 2
                    elif src[i2] == '\n':
                        break
                    else:
                        i2 += 1
                q3blocks.append((i1, i2))
                i = i2 + 1
            elif src[i1] == '#':
                i = find(('\n',),  i1 + 1)  + 1
                comments.append(i1)
            else:
                i = i1
        q3blocks.append((N,N))
        comments.append(N)
        #@+node:vitalije.20180529151834.4: *4* isq3
        def isq3(i, x):
            while q3blocks[0][1] < i:
                q3blocks.pop(0)
            a, b = q3blocks[0]
            return a <= i <= b
        #@+node:vitalije.20180529151834.5: *4* isindent
        def isindent(i, x, cmi):
            a, b = q3blocks[0]
            return (a > i or b < cmi) and \
                x[:cmi - i].rstrip().endswith((':', ':\n'))
        #@+node:vitalije.20180529151834.6: *4* mkoneline
        pos = [0]
        def mkoneline(i, x):
            a = pos[0]
            cl = not isq3(a, x)
            while comments[0] < a:
                comments.pop(0)
            pos[0] = a + len(x)
            cmi = min(comments[0], pos[0])
            isInd = isindent(a, x, cmi)
            cpos = cmi - a
            xlstr = x.lstrip()
            ind = len(x) - len(xlstr) if xlstr else 0
            if not xlstr:
                x = '\n'
                cpos = 1
            return Line(i, a, pos[0], ind, cl, isInd, cpos, x)
        #@-others
        return [mkoneline(i, x) for i, x in enumerate(lines)]
    slines = srcLines(src)
    #@+node:vitalije.20180529152008.1: *3* hname
    def hname(x):
        if x.txt.startswith('def ', x.ind):
            return x.txt[x.ind+4:].split('(',1)[0]
        elif x.txt.startswith('class ', x.ind):
            s = x.txt[x.ind+6:] + '('
            i = min(s.find('('), s.find(':'))
            return s[:i]
        else:
            return x.txt[x.ind:]
    #@+node:vitalije.20180529150801.1: *3* init import
    ind = 0
    attrs = {}
    attrs[rgnx] = [[], [], '', '']
    #@+node:vitalije.20180529150815.1: *4* add_new_node
    def add_new_node(vr):
        gnx = new_gnx()
        attrs[vr][1].append(gnx)
        attrs[gnx] = [[vr], [], '', '']
        return gnx
    #@+node:vitalije.20180529151422.1: *3* next_start
    def next_start(x, slines):
        while not x.isInd:
            x = slines[x.ln + 1]
        return x
    #@+node:vitalije.20180529151433.1: *3* first_in_block
    def first_in_block(x, slines):
        y = slines[x.ln + 1]
        while not y.cl or y.cpos - y.ind <= 1:
            y = slines[y.ln + 1]
        return y
    #@+node:vitalije.20180529151449.1: *3* next_statement
    def next_statement(x, lev, slines):
        y = slines[x.ln + 1]
        N = slines[-1].ln
        while y.ln < N:
            while not y.cl or y.cpos - y.ind <= 1:
                if y.ln < N:
                    y = slines[y.ln + 1]
                else:
                    break
            if y.ind <= lev:
                break
            elif y.ln < N:
                y = slines[y.ln + 1]
        return y
    #@+node:vitalije.20180529151407.1: *3* inside_block
    def inside_block(x, slines):
        y = next_start(x, slines)
        y = first_in_block(y, slines)
        z = next_statement(y, x.ind, slines)
        return y, z.ln - x.ln
    #@+node:vitalije.20180529151523.1: *3* mkvnode
    def mkvnode(vr, x, sz, tl, slines):
        v = new_gnx()
        attrs[v] = [[vr], [], hname(x), '']
        attrs[vr][1].append(v)
        i = x.ln + sz
        N = slines[-1].ln
        f = lambda y:y.cl and y.cpos - y.ind <= 1
        f1 = lambda y:y.ln < N and not(y.txt.strip())
        while f(slines[i-1]):i -= 1
        while f1(slines[i]):i += 1
        return i, v
    #@+node:vitalije.20180529151243.1: *3* nodes_from_lines
    def nodes_from_lines(vr, ind, slines, a, b):
        tl = 0
        res = {}
        filt = lambda x: (
            x.cl and
            x.txt.startswith(('def ', 'class '), ind) and
            x.cpos - x.ind > 1)
        filt2 = lambda x: ':' in x.txt and x.txt.rsplit(':', 1)[-1].strip()
        fclines = [x for x in slines[a:b] if filt(x)]
        if not fclines:
            res[vr] = [None, tl, b, b, b, ind, None]
            return res
        for x in fclines:
            if filt2(x):
                fl, sz = x, 1
            else:
                fl, sz = inside_block(x, slines)
            if vr not in res:
                res[vr] = [None, tl, x.ln, x.ln+sz, b, ind, None]
                tl = x.ln
            tl2, v = mkvnode(vr, x, sz, tl, slines)
            res[v] = [x, tl, tl2, tl2, tl2, ind, fl]
            tl = tl2
        if tl < len(slines):
            res[vr][3] = tl
        return res
    #@+node:vitalije.20180529152034.1: *3* add_lines
    def add_lines(slines, blines, ind):
        efind = ind
        for x in slines:
            if x.ind != 0 or x.cpos != 1:
                if x.ind < efind:
                    efind = x.ind
                    blines.append('@setindent %d\n'%x.ind)
                elif x.ind >= ind and efind != ind:
                    efind = ind
                    blines.append('@setindent %d\n'%ind)
            blines.append(x.txt[efind:])
        if efind != ind:
            blines.append('@setindent %d\n'%ind)
    #@+node:vitalije.20180529151122.1: *3* do import
    res = nodes_from_lines(rgnx, ind, slines, 0, len(slines))
    todo = set(res.keys())
    while todo:
        gnx = todo.pop()
        x, a, b, c, d, ind, fl = res[gnx]
        if b - a > 30 and x and x.txt.startswith('class ', x.ind):
            r1 = nodes_from_lines(gnx, fl.ind, slines, fl.ln, b)
            b, c, d = r1.pop(gnx)[2:5]
            res[gnx] = x, a, b, c, d, ind, fl
            res.update(r1)
            todo.update(r1.keys())
    #@+node:vitalije.20180529151151.1: *3* fill lines
    for gnx in res:
        x, a, b, c, d, ind, fl = res[gnx]
        atr = attrs[gnx]
        if x:
            atr[2] = hname(x)
        blines = []
        if fl:
            add_lines(slines[a:fl.ln], blines, ind)
            add_lines(slines[fl.ln:b], blines, ind)
            if d >= c > b:
                blines.append((' '* (fl.ind - ind)) + '@others\n')
                add_lines(slines[c:d], blines, ind)
        else:
            add_lines(slines[a:b], blines, 0)
            if d >= c > b:
                blines.append('@others\n')
                add_lines(slines[c:d], blines, 0)

        atr[3] = ''.join(blines)
    #@-others
    def viter(gnx, lev0):
        s = [1]
        ps, chn, h, b = attrs[gnx]
        mnode = (gnx, h, b, lev0, s, ps, chn)
        yield mnode
        for ch in chn:
            for x in viter(ch, lev0 + 1):
                s[0] += 1
                yield x
    return nodes2treemodel(tuple(viter(rgnx, 0)))
#@+node:vitalije.20180529163709.1: ** p_to_autolines
def p_to_autolines(ltmdata, pos):
    '''Returns an iterator of lines generated by at-auto node p.
       It respects at-others and at-setindent directives. Does
       not expand section references and treats section nodes
       as ordinary ones, puts their content among other nodes.
       
       yields tuples of (line, ni, lln) where 
            line is text content
            ni - index of currently outputing node
            lln - local line number, i.e. line number in the
                  currently outputing body
    '''
    (positions, nodes, attrs, levels, expanded, marked) = ltmdata

    NI = [positions.index(pos)]
    #@+others
    #@+node:vitalije.20180529163709.2: *3* all_lines
    def all_lines(gnx, ind):
        mNI = NI[0]
        h, b, ps, chn, sz = attrs[gnx]
        lines = b.splitlines(True)
        for i, line in enumerate(lines):
            #@+others
            #@+node:vitalije.20180529163709.3: *4* at - others
            if '@others' in line:
                sline = line.lstrip()
                ws = len(line) - len(sline)
                if sline == '@others\n':
                    for ch in chn:
                        NI[0] += 1
                        for x in all_lines(ch, ind + ws):
                            yield x
                    continue
            #@+node:vitalije.20180529163709.4: *4* at - setindent
            if line.startswith('@setindent '):
                ind = int(line[11:].strip())
                continue
            #@+node:vitalije.20180529163709.5: *4* directives
            if line.startswith((
                '@killcolor\n',
                '@nocolor\n',
                '@language ')):continue
            #@-others
            if ind: line = (' '*ind) + line
            yield line, mNI, i
    #@-others
    return all_lines(nodes[NI[0]], 0)
#@+node:vitalije.20180630121324.1: ** Search and replace
#@+others
#@+node:vitalije.20180630121402.1: *3* FindArgs
FindArgs = namedtuple('FindArgs', 't_find t_replace flags')

def fa_to_string(fa):
    res = ['%s:%s'%(x, bool(fa.flags & FA_FLAGS[x])) for x in FA_FLAGS]
    res.append('t_find:' + fa.t_find)
    res.append('t_replace:' + fa.t_replace)
    return '\n'.join(res)

def str_to_fa(s):
    t_find = ''
    t_replace = ''
    kw = {}
    for line in s.splitlines():
        if ':' not in line:continue
        x,y = line.split(':')
        if x == 't_find':
            t_find = y
        elif x == 't_replace':
            t_replace = y
        elif x not in FA_FLAGS:
            continue
        elif y in ('True', 'yes', 'on', '1', 'true'):
            y = True
        else:
            y = False
        kw[x] = y
    return FindArgs(t_find, t_replace, fa_flags(**kw))

#@+node:vitalije.20180630121411.1: *3* FA_FLAGS
FA_FLAGS = dict((x, (1<<i)) for i,x in enumerate((
    'ignore_case', 'node_only', 'pattern_match',
    'search_headline', 'search_body', 'suboutline_only',
    'mark_changes', 'mark_finds', 'reverse', 'wrap',
    'whole_word')))
#@+node:vitalije.20180630121328.1: *3* fa_flags
def fa_flags(**kw):
    res = kw.get('flags', 24)
    for k, v in kw.items():
        if k in FA_FLAGS:
            f = FA_FLAGS[k]
            res = res | f if v else res & ~f
    return res
#@+node:vitalije.20180630121331.1: *3* find_args
def find_args(f, r, flags=0, **kw):
    return FindArgs(f, r, fa_flags(flags=flags, **kw))
#@+node:vitalije.20180630124123.1: *3* build_search_pattern
def build_search_pattern(txt, flags):
    if flags & FA_FLAGS['pattern_match']:
        ptxt = txt
    else:
        ptxt = re.escape(txt)
    if flags & FA_FLAGS['whole_word']:
        ptxt = '\\b' + ptxt + '\\b'
    if flags & FA_FLAGS['ignore_case']:
        reflags = re.I
    else:
        reflags = 0
    return re.compile(ptxt, reflags)
#@+node:vitalije.20180630131343.1: *3* SearchState
class SearchState:
    def __init__(self, fargs, pos, sel_start, sel_end, in_head):
        self.t_find, self.t_replace, self.flags = fargs
        self.start_pos = pos
        self.start_gnx = None
        self._start_ind = 0
        self.sel_start = sel_start
        self.sel_end = sel_end
        self.in_head = in_head
        self.pattern = build_search_pattern(self.t_find, self.flags)
        self.pos_index = 0
        self.curr_pos = pos
        self.did_wrap = False
        self._last_match = None

    def set_args(self, fargs):
        if self.flags != fargs.flags:
            f = fargs.flags & FA_FLAGS['reverse']
            f = f or not (fargs.flags & FA_FLAGS['search_headline'])
            self.in_head = not f
        self.t_find, self.t_replace, self.flags = fargs
        self.pattern = build_search_pattern(self.t_find, self.flags)

    #@+others
    #@+node:vitalije.20180630131511.1: *4* curr_index
    def curr_index(self, ltmdata):
        if ltmdata.positions[self.pos_index] != self.curr_pos:
            self.pos_index = ltmdata.positions.index(self.curr_pos)
        if self.curr_pos == self.start_pos:
            self.start_gnx = ltmdata.nodes[self.pos_index]
        return self.pos_index
    #@+node:vitalije.20180630131516.1: *4* next_index
    def next_index(self, ltmdata):
        posi = self.curr_index(ltmdata)
        if self.flags & FA_FLAGS['node_only']:
            return
        N = len(ltmdata.positions)
        if self.flags & FA_FLAGS['reverse']:
            posi -= 1
            if posi <= 0 and self.flags & FA_FLAGS['wrap']:
                posi += N
                self.did_wrap = True
        else:
            posi += 1
            if posi >= N and self.flags & FA_FLAGS['wrap']:
                posi = 1
                self.did_wrap = True
        if self.flags & FA_FLAGS['suboutline_only']:
            N = self.start_index(ltmdata) + ltmdata.attrs[self.start_gnx].size
        self.pos_index = posi
        if 0 < posi < N:
            self.curr_pos = ltmdata.positions[posi]
            if self.flags & FA_FLAGS['reverse']:
                txt = self.curr_txt(ltmdata)
                self.sel_start = len(txt)
                self.sel_end = self.sel_start
            else:
                self.sel_start = 0
                self.sel_end = 0
            return posi
        else:
            return None
    #@+node:vitalije.20180630131522.1: *4* curr_txt
    def curr_txt(self, ltmdata):
        i = self.curr_index(ltmdata)
        gnx = ltmdata.nodes[i]
        if self.in_head and (self.flags & FA_FLAGS['search_headline']):
            return ltmdata.attrs[gnx].h
        elif not self.in_head and (self.flags & FA_FLAGS['search_body']):
            return ltmdata.attrs[gnx].b
        else:
            return ''
    #@+node:vitalije.20180630131526.1: *4* advance
    def advance(self, ltmdata):
        if self.flags & FA_FLAGS['reverse']:return self.advance_reverse(ltmdata)
        if self.in_head and self.flags & FA_FLAGS['search_body']:
            self.in_head = False
            return self.pos_index
        elif not self.in_head and self.flags & FA_FLAGS['search_headline']:
            self.in_head = True
        return self.next_index(ltmdata)

    def advance_reverse(self, ltmdata):
        if not self.in_head and self.flags & FA_FLAGS['search_headline']:
            self.in_head = True
            txt = self.curr_txt(ltmdata)
            self.sel_start = self.sel_end = len(txt)
            return self.pos_index
        if self.flags & FA_FLAGS['search_body']:
            self.in_head = False
        return self.next_index(ltmdata)
    #@+node:vitalije.20180630131536.1: *4* get_match
    def get_match(self, txt, ins):
        if self.flags & FA_FLAGS['reverse']:
            ms = list(self.pattern.finditer(txt[:ins]))
            return ms and ms[-1] or None
        else:
            return self.pattern.search(txt, ins)
    #@+node:vitalije.20180630131532.1: *4* find_next
    def find_next(self, ins, ltmdata):
        i = self.curr_index(ltmdata)
        txt = self.curr_txt(ltmdata)
        m = self.get_match(txt, ins)
        while not m:
            if self.did_wrap and self.start_pos == self.curr_pos:
                break
            if not self.advance(ltmdata):
                break
            txt = self.curr_txt(ltmdata)
            if self.flags & FA_FLAGS['reverse']:
                m = self.get_match(txt, self.sel_start)
            else:
                m = self.get_match(txt, self.sel_end)
        if m:
            self._last_match = m
            self.sel_start = m.start()
            self.sel_end = m.end()
        else:
            self._last_match = None
        return m
    #@-others
    def start_index(self, ltmdata):
        if ltmdata.positions[self._start_ind] != self.start_pos:
            self._start_ind = ltmdata.positions.index(self.start_pos)
        return self._start_ind
#@-others
#@-others
#@-leo
