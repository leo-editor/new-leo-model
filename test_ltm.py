'''this script should test LeoTreeModel without parPos'''
from collections import defaultdict
from leoDataModel import (LeoTreeModel, copy_ltmdata, load_leo_full, ltm_from_derived_file,
    new_gnx, parPosIter, nodes2treemodel)
import hypothesis.strategies as st
from hypothesis.stateful import (RuleBasedStateMachine, rule, initialize, invariant,
    precondition, run_state_machine_as_test)
from hypothesis import settings, PrintSettings, Verbosity, Phase
import string
import time
import sys
def svgexp(pref, findex, ltm):
    _p2txt = {}
    for i, p in enumerate(ltm.data.positions):
        if p not in _p2txt:
            _p2txt[p] = 'P%d'%i

    def p2txt(x):
        if x not in _p2txt:
            _p2txt[x] = 'P%d'%len(_p2txt)
        return _p2txt[x]
    #print(','.join(_p2txt[p] for p in ltm.data.positions))

    def mk_gnx2txt():
        _gnx2txt = {}
        N = len(ltm.data.attrs)
        n1 = 1
        if N > 26:
            n1 = 2
        if N > 26 * 26:
            n1 = 3
        def g2a(i):
            i, a2 = divmod(i, 26 * 26)
            a1, a0 = divmod(i, 26)
            return (chr(65 + a2) + chr(65 + a1) + chr(65 + a0))[:n1]
        for i, gnx in enumerate(sorted(ltm.data.attrs.keys())):
            _gnx2txt[gnx] = g2a(i)

        def gnx2txt(x):
            if x not in _gnx2txt:
                _gnx2txt[x] = g2a(len(_gnx2txt))
            return _gnx2txt[x]
        return gnx2txt
    g2txt = mk_gnx2txt()

    def row(tr, i):
        if len(tr.positions) > i:
            x = tr.positions[i]
            gnx = tr.nodes[i]
            isMarked = gnx in tr.marked
            apos = tuple('P%d'%x for x,y in enumerate(tr.nodes) if y == gnx)
            h, b, ps, chn, sz = tr.attrs[gnx]
            achn = [g2txt(x) for x in chn]
            aps = [g2txt(x) for x in ps]
            lev = tr.levels[i]
            if lev < 2:
                pp = 'P0'
            else:
                pp = 'P%d'%tr.levels.rfind(lev-1, 0, i)
            return p2txt(x), pp, lev, sz, h, apos, g2txt(gnx), achn, aps, isMarked
        return '...', '...', -1, 0, '...nista...', tuple(), '...', tuple(), tuple(), False

    def addrect(x, y, w, h, svg):
        svg.append('<svg:rect x="%f" y="%f" width="%f" height="%f" style="%s"/>'%(
            x, y, w, h, 'fill:#ffffff;stroke:#000000;stroke-width:0.25'))

    def addtext(x, y, t, sz, svg, isRed):
        t = t.replace('&', '&amp;')
        t = t.replace('<', '&lt;')
        t = t.replace('>', '&gt;')
        f = '#ff0000' if isRed else '#000000'
        svg.append(
            ('<svg:text style="font-family:Verdana,Arial;font-size:%fpx;fill:%s;fill-opacity:1;text-align:center;text-anchor:middle;" '%(sz, f)) +
            ('x="%f" y="%f">'%(x+10, y)) +
            ('<svg:tspan x="%f" y="%f">%s</svg:tspan></svg:text>'%(x+10, y, t)))

    def addtext_l(x, y, t, sz, svg, isRed):
        t = t.replace('&', '&amp;')
        t = t.replace('<', '&lt;')
        t = t.replace('>', '&gt;')
        f = '#ff0000' if isRed else '#000000'
        svg.append(
            ('<svg:text style="font-family:Verdana,Arial;font-size:%fpx;fill:%s;fill-opacity:1;" '%(sz, f)) +
            ('x="%f" y="%f">'%(x, y)) +
            ('<svg:tspan x="%f" y="%f">%s</svg:tspan></svg:text>'%(x, y, t)))

    def addTreeG(tr, a, b, x, y, res, reds):
        res.append('<svg:g transform="translate(%f,%f)">'%(x, y))
        h = 10
        res.append('<svg:rect x="0" y="0" width="500" height="%f" style="%s"/>'%(
            (h+2)*(b - a + 1),
            'fill:#ffffee;stroke:#000000;stroke-width:0.25'))
        addtext(2, 4, 'positions', 4, res, False)
        addtext(22, 4, 'nodes', 4, res, False)
        addtext(44, 4, 'levels', 4, res, False)
        addtext(66, 4, 'size', 4, res, False)
        for i in range(a, b):
            ri = 0.5 + i - a
            p, pp, lev, sz, head, allpos, agnx, chn, pnts, isMarked = row(tr, i)
            addrect(0, ri * (h + 2), 20, h, res)
            if isMarked:
                addrect(14, ri * (h + 2) + 4, 4, h-8, res)
            addrect(22, ri * (h + 2), 20, h, res)
            addrect(44, ri * (h + 2), 20, h, res)
            addrect(66, ri * (h + 2), 20, h, res)

            addtext(0,  8+ri * (h + 2), p, 8, res, (i,0) in reds)
            addtext(22, 8+ri * (h + 2), agnx, 8, res, (i,1) in reds)
            addtext(44, 8+ri * (h + 2), str(lev), 8, res, (i,2) in reds)
            addtext(66, 8+ri * (h + 2), str(sz), 8, res, (i,3) in reds)
            addtext_l(88 + (lev * 14), 8+ri * (h + 2), head, 8, res, (i,4) in reds)
            sss = ', '.join(allpos) + '   children:[' + ' '.join(chn) + ']  / parents:[' + ' '.join(pnts) + ']'
            addtext_l(220, 8 + ri * (h + 2), sss, 6, res, False)
        res.append('</svg:g>')

    svgheader = '\n'.join(x.lstrip() for x in '''<?xml version="1.0" encoding="UTF-8" ?>
    <svg
       xmlns:dc="http://purl.org/dc/elements/1.1/"
       xmlns:cc="http://creativecommons.org/ns#"
       xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
       xmlns:svg="http://www.w3.org/2000/svg"
       xmlns="http://www.w3.org/2000/svg"
       xmlns:xlink="http://www.w3.org/1999/xlink"
       xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
       xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
       width="400mm"
       height="205mm"
       viewBox="0 0 400 255"
       version="1.1"
       id="svg8">'''.split('\n'))
    def mksvg(cont, sx, sy):
        res = [svgheader,'<svg:g transform="scale(%f, %f)">'%(sx, sy), cont, '</svg:g></svg>']
        return ''.join(res)

    def mkpic(s, n, t1, t2, a, b, inserted=tuple(), deleted=tuple()):
        fname = 'tmp/%s-%d.svg'%(s, n)
        with open(fname, 'w', encoding="utf8") as out:
            svg = []
            addTreeG(t1, a, b, 0, 0, svg, [])
            if t2:
                _red = reds(a, b, t1, t2, inserted, deleted)
                addTreeG(t2, a, b, 250, 0, svg, _red)
            cont = ''.join(svg)
            sc = 250/13 / max(20, len(t1.positions))
            out.write(mksvg(cont, sc,sc))

    def reds(a, b, ta, tb, inserted, deleted):
        di = 0
        res = []
        for i in range(a, b):
            if i in inserted:
                di += 1
                res.extend(((i,0),(i,1),(i,2),(i,3),(i,4)))
                continue
            elif i in deleted:
                di -= 1
                continue
            j = i + di
            r1 = row(ta, i)
            r2 = row(tb, j)
            for k in range(5):
                if r1[k] != r2[k]:
                    res.append((j,k))
        return tuple(res)

    mkpic(pref, findex, ltm.data, None, 0, ltm.size, [], [])
    #print(findex, 'svg')

allgnxes = [0, tuple('gnx.%04d'%i for i in range(5000))]
def ngnx():
    a, b = allgnxes
    algnxes[0] = (a + 1)%len(b)
    return b[a]

def mk_tree_strategy(min_size=30, max_size=80):
    def level_size(total, lev):
        levdivs = (5/6, 1/7, 1/10, 1/20, 1/40, 1/60, 1/80, 1/100)
        assert lev < len(levdivs), 'max lev allowed %d'%(len(lendivs))
        return max(1, int(total*levdivs[lev]))

    def gnx_from_level(lsizes, pool):
        def mkone():
            i = 0
            j = 0
            a, b = 0, lsizes[0]
            while j < len(pool):
                x = pool[j]
                j += 1
                if j > b:
                    i += 1
                    a = b
                    if i < len(lsizes):
                        b += lsizes[i]
                    else:
                        b = len(pool)
                if i == 0:
                    yield st.tuples(st.just(x), st.builds(list))
                else:
                    yield st.tuples(st.just(x), st.lists(st.sampled_from(pool[:a]), min_size=1, max_size=5))
        return st.tuples(st.just(lsizes), st.tuples(*mkone()))

    def mk_tree_model(row):
        lsizes, ns = row
        cdict = dict(ns)
        cdict['hidden-root-vnode-gnx'] = [x[0] for x in ns[-lsizes[-1]:]]
        pdict = defaultdict(list)
        heads = {'hidden-root-vnode-gnx':'root-head'}
        bodies = {'hidden-root-vnode-gnx':'root-body'}
        for x,y in ns:
            heads[x] = 'head-%d'%len(heads)
            bodies[x] = 'body-%d'%len(bodies)
        seen = set()
        def addlink(x):
            if x in seen: return
            seen.add(x)
            for y in cdict[x]:
                pdict[y].append(x)
                addlink(y)
        addlink('hidden-root-vnode-gnx')
        def viter(v, lev0):
            s = [1]
            mnode = (v, heads[v], bodies[v], lev0, s, pdict[v], cdict[v])
            yield mnode
            for ch in cdict[v]:
                for x in viter(ch, lev0 + 1):
                    s[0] += 1
                    yield x
        return nodes2treemodel(tuple(viter('hidden-root-vnode-gnx', 0)))

    gnxst = st.builds(ngnx)
    levsizes = st.tuples(
        st.integers(min_value=min_size, max_value=max_size), 
        st.integers(min_value=2, max_value=5)
    ).map(lambda x:tuple(level_size(x[0], i) for i in range(x[1])))
    nodepool = levsizes.map(lambda x:(x, allgnxes[1][:sum(x)]))
    return nodepool.flatmap(lambda x:gnx_from_level(x[0], x[1])).map(mk_tree_model)
class LTM_rule_machine(RuleBasedStateMachine):
    def int2pos(self, pi):
        ps = self.model.data.positions
        pi1 = abs(pi) % (len(ps)-1) + 1
        if self.mdebug:print('%d -> %d'%(pi, pi1))
        return ps[pi1]

    def int2int(self, pi):
        N = len(self.model.data.positions) - 1
        pi1 = 1 + (abs(pi) % N)
        if self.mdebug:print('%d -> %d'%(pi, pi1))
        return pi1

    def replace_ok(self, p1, p2):
        (positions, nodes, attrs, levels, expanded, marked) = self.model.data

        p1i = abs(p1) % (len(nodes)-1) + 1
        p2i = abs(p2) % (len(nodes)-1) + 1
        if self.mdebug:print('(%d,%d)->(%d,%d)'%(p1,p2,p1i,p2i))
        sz = attrs[nodes[p1i]][4]
        if nodes[p2i] not in nodes[p1i:p1i+sz]:
            return self.model.subtree(positions[p1i]), nodes[p2i]
        else:
            return (None, None)

    def err(self):
        e = self.errnum
        self.errnum += 1
        pref = 'ltm%d'%self.examplenum
        svgexp(pref, e, self.model)

    @initialize(tr=mk_tree_strategy())
    def init(self, tr):
        self.model = tr
        self.mdebug = False
        if not hasattr(self, 'examplenum'):
            self.examplenum = 1
        else:
            self.examplenum += 1
        with open('tmp\\trdata%03d.bin'%self.examplenum, 'wb') as out:
            out.write(tr.to_bytes())

        self.errnum = 1

    def pr_tr(self):
        td = self.model.data
        def r():
            for x, lev in zip(td.nodes, td.levels):
                if self.model.isClone(x):
                    cl = 'C'
                else:
                    cl = ' '
                yield '%s[%s]%s'%('- - '*lev, cl, self.model.head(x))
        print('\n'.join(r()))

    @precondition(lambda self:hasattr(self, 'model'))
    @invariant()
    def positions_are_unique(self):
        n1 = len(set(self.model.data.positions))
        n2 = len(self.model.data.positions)
        if n1 != n2:self.err()
        assert n1 == n2

    @invariant()
    def gnxes_are_correct(self):
        if not hasattr(self, 'model'): return
        td = self.model.data
        zlev = zip(range(len(td.nodes)), td.nodes, td.levels)

        for i1, x, lev in zlev:
            chn = td.attrs[x][3]
            for j, i, gnx in self.model._child_iterator(i1):
                assert gnx == chn[j], 'wrong order of children %r'%((i1, x, chn, j, i, gnx),)
                assert td.levels[i] == lev + 1, 'wrong level of direct child %r'%((x,i1, i, j, gnx),)
                assert chn.count(gnx) == td.attrs[gnx][2].count(x), 'wrong number of parent links'

    @invariant()
    def marked_contains_no_extra_gnxes(self):
        if not hasattr(self, 'model'): return
        marked = self.model.data.marked
        attrs = self.model.data.attrs
        for x in marked:
            assert x in attrs, 'marked contains gnx which does not exist in attrs %r'%x
    @rule(pi=st.integers())
    def mark_set(self, pi):
        pos = self.int2pos(pi)
        self.model.set_mark(pos)

    @rule(pi=st.integers())
    def mark_clear(self, pi):
        pos = self.int2pos(pi)
        self.model.clear_mark(pos)

    @precondition(lambda self:self.model.size < 300)
    @rule(pi=st.integers())
    def clone_marked(self, pi):
        pos = self.int2pos(pi)
        self.model.selectedPosition = pos
        self.model.clone_marked(new_gnx)

    @precondition(lambda self:self.model.size < 300)
    @rule(pi=st.integers())
    def delete_marked(self, pi):
        pos = self.int2pos(pi)
        self.model.selectedPosition = pos
        self.model.clone_marked(new_gnx)

    @rule(p1=st.integers(), p2=st.integers())
    def replace_node(self, p1, p2):
        tr, gnx = self.replace_ok(p1, p2)
        if tr:
            tr.change_gnx(tr.data.nodes[0], gnx)
            self.model.replace_node(tr)

    @rule(pi=st.integers())
    def sort_children(self, pi):
        self.model.sort_children(self.int2pos(pi))

    @rule(pi=st.integers())
    def toggle_node(self, pi):
        self.model.toggle(self.int2pos(pi))

    @precondition(lambda self:self.model.size < 300)
    @rule(pi=st.integers())
    def clone_node(self, pi):
        self.model.clone_node_i(self.int2int(pi))

    @rule(pi=st.integers())
    def move_node_down(self, pi):
        self.model.move_node_down(self.int2pos(pi))

    @rule(pi=st.integers())
    def move_node_up(self, pi):
        self.model.move_node_up(self.int2pos(pi))

    @rule(pi=st.integers())
    def move_node_left(self, pi):
        self.model.dedent_node(self.int2pos(pi))

    @rule(pi=st.integers())
    def move_node_right(self, pi):
        self.model.indent_node(self.int2pos(pi))

    @rule(pi=st.integers())
    def promote_node(self, pi):
        self.model.promote(self.int2pos(pi))

    @rule(pi=st.integers())
    def demote_node(self, pi):
        self.model.promote_children(self.int2pos(pi))

    @rule(pi=st.integers())
    def delete_node(self, pi):
        self.model.delete_node(self.int2pos(pi))

def mkstate():
    ltm = LeoTreeModel.from_bytes(open('tmp/trdata001.bin', 'rb').read())
    state = LTM_rule_machine()
    state.examplenum = 1
    state.errnum = 1
    state.model = ltm
    state.mdebug = True
    return state

def do_example_1():
    import pdb
    state = mkstate()
    state.err() # 1
    state.promote_node(pi=0) # 1
    state.err() # 2
    state.toggle_node(pi=256) # 1
    state.err() # 3
    state.move_node_left(pi=-29091) # 4
    state.err() # 4
    state.move_node_left(pi=-8545) # 2
    state.err() # 5
    state.move_node_up(pi=66) # 3
    state.err() # 6
    state.clone_node(pi=-3774) # 7
    state.err() # 7
    state.clone_marked(pi=29926) # 2
    state.err() # 8
    state.clone_marked(pi=-4579) # 8
    state.err() # 9
    state.mark_clear(pi=-29222) # 9
    state.err() # 10
    state.move_node_down(pi=-10) # 2
    state.err() # 11
    state.clone_node(pi=6352) # 8
    state.err() # 12
    state.demote_node(pi=-24) # 5
    state.err() # 13
    state.mark_clear(pi=61) # 2
    state.err() # 14
    state.mark_clear(pi=-31100) # 1
    state.err() # 15
    state.promote_node(pi=27354) # 5
    state.err() # 16
    state.mark_set(pi=13875) # 6
    state.err() # 17
    state.mark_clear(pi=21021) # 2
    state.err() # 18
    state.replace_node(p1=-122, p2=72121929359983009) # (3,10)
    state.err() # 19
    state.demote_node(pi=-6) # 7
    state.err() # 20
    state.replace_node(p1=2907, p2=120) # (8,1)
    state.err()

with settings(max_examples=90, stateful_step_count=1000,
            #verbosity=Verbosity.debug,
            max_shrinks=200,
            print_blob=PrintSettings.ALWAYS,
            phases=[Phase.explicit,
                Phase.reuse, 
                Phase.generate]) as S:
    LTM_Test = LTM_rule_machine.TestCase
    LTM_Test.settings = S
if __name__ == '__main__':
    do_example_1()
