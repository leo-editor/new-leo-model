#@+leo-ver=5-thin
#@+node:vitalije.20180514194301.1: * @file miniTkLeo.py
#@@language python
G = None
#@+others
#@+node:vitalije.20180518115138.1: ** imports
from tkinter import Canvas, Tk, scrolledtext, PanedWindow, PhotoImage, IntVar
from tkinter import font
import threading
import queue
import time
#from tkinter import ttk
from leoDataModel import (
         LeoTreeModel,
         load_leo,
         load_external_files,
         ltm_from_derived_file,
         p_to_lines,
         paths,
         SearchState,
         FindArgs, fa_to_string,
         str_to_fa,
         FA_FLAGS
    )
import sys
import os
from box_images import boxKW, plusnode, minusnode
assert ltm_from_derived_file
assert p_to_lines
assert paths
#@+node:vitalije.20180516112439.1: ** anglestr
anglestr = lambda x: ''.join(('<<', x,
    '>>'))

#@+node:vitalije.20180514223632.1: ** main
def main(fname):
    tstart = time.monotonic()
    #@+others
    #@+node:vitalije.20180518114840.1: *3* 1. load xml
    ltm = load_leo(fname)
    ltm.selectedPosition = ltm.data.positions[1]
    ltmbytes = ltm.to_bytes()
    #@+node:vitalije.20180518114847.1: *3* 2. create app
    app = Tk()
    app.columnconfigure(0, weight=1)
    app.rowconfigure(0, weight=1)
    #@+node:vitalije.20180518114857.1: *3* 3. adjust fonts
    font.nametofont('TkFixedFont').config(size=18)
    font.nametofont('TkTextFont').config(size=18)
    font.nametofont('TkDefaultFont').config(size=18)
    #@+node:vitalije.20180518114953.1: *3* 4. create gui
    f1 = PanedWindow(app, orient='horizontal', width=800, height=600,
        sashrelief='ridge', sashwidth=4)
    f1.grid(row=0, column=0, sticky="nesw", )
    f2 = PanedWindow(f1, orient='vertical')
    canvW = Canvas(f2, bg='#113333')
    f2.add(canvW)
    logW = scrolledtext.ScrolledText(f2, bg='#223399', fg='#cccc99',
        font=font.nametofont('TkDefaultFont'))
    f2.add(logW)
    bodyW = makeBodyW(f1)
    f1.add(f2)
    f1.add(bodyW)
    #@+node:vitalije.20180518115000.1: *3* 5. f_later
    def f_later():
        f1.sash_place(0, 270, 1)
        f2.sash_place(0, 1, 350)
        app.geometry("800x600+720+50")
        app.wm_title(fname)
        app.after_idle(update_model)
    #@+node:vitalije.20180518115003.1: *3* 6. loadex
    def loadex():
        ltm2 = LeoTreeModel.from_bytes(ltmbytes)
        loaddir = os.path.dirname(fname)
        load_external_files(ltm2, loaddir)
        G.q.put(ltm2)
    #@+node:vitalije.20180518115010.1: *3* 7. update_model
    def update_model():
        try:
            m = G.q.get(False)
            ltm.data = m.data
            draw_tree(G.tree, ltm)
            tend = time.monotonic()
            t1 = (tend - tstart)
            logW.insert('end', 'External files loaded in %.3fs\n'%t1)
        except queue.Empty:
            app.after(100, update_model)
    #@+node:vitalije.20180518115038.1: *3* 8. start loading thread
    threading.Thread(target=loadex, name='externals-loader').start()
    app.after_idle(f_later)
    #@-others
    return bunch(
        ltm=ltm,
        app=app,
        tree=canvW,
        body=bodyW,
        log=logW,
        q=queue.Queue(1),
        topIndex=IntVar(),
        search=SearchState(str_to_fa(''), ltm.data.positions[1], 0, 0, False))
#@+node:vitalije.20180515145134.1: ** makeBodyW
def makeBodyW(parent):
    bw = scrolledtext.ScrolledText(parent, font='Courier 18', undo=False)
    bw._orig = bw._w + '_orig'
    bw.tk.call('rename', bw._w, bw._orig)
    def proxycmd(cmd, *args):
        mod = cmd in ('insert', 'delete', 'replace')
        if mod:
            prep_undo()
        result = bw.tk.call(bw._orig, cmd, *args)
        if mod:
            bw.event_generate('<<'
                'TextModified>>')
        return result
    bw.tk.createcommand(bw._w, proxycmd)
    bw.setBodyTextSilently = lambda x:bw.tk.call(bw._orig, 'replace', '1.0', 'end', x)
    return bw
#@+node:vitalije.20180515153501.1: ** connect_handlers
def connect_handlers():
    bw = G.body
    tree = G.tree
    ltm = G.ltm
    topIndex = G.topIndex

    def speedtest(x):
        import timeit
        pos = ltm.data.positions[-4]
        def f1():
            ltm.promote(pos)
            draw_tree(tree, ltm)
            ltm.invalidate_visual()
            ltm.promote_children(pos)
            draw_tree(tree, ltm)
            ltm.invalidate_visual()
        def f2():
            ltm.move_node_down(pos)
            draw_tree(tree, ltm)
            ltm.invalidate_visual()
            ltm.move_node_up(pos)
            draw_tree(tree, ltm)
            ltm.invalidate_visual()

        t1 = timeit.timeit(f1, number=100)/100*1000
        t2 = timeit.timeit(f2, number=100)/100*1000
        G.log.insert('end', 'demote/promote average: %.1fms\n'%t1)
        G.log.insert('end', 'up/down average: %.1fms\n'%t2)
    def traverse_speed(x):
        def tf():
            nodes = ltm.data.nodes
            attrs = ltm.data.attrs
            levels = ltm.data.levels
            res = []
            for i, gnx in enumerate(nodes):
                res.append(('----'*levels[i]) + attrs[gnx].h)
            return '\n'.join(res)
        import timeit
        t1 = timeit.timeit(tf, number=100)*1000/100
        G.log.insert('end', 'Average: %.1fms\n'%t1)
    #@+others
    #@+node:vitalije.20180516131732.1: *3* prev_node
    def prev_node(x):
        gnx = ltm.select_prev_node()
        if gnx:
            G.body.setBodyTextSilently(ltm.selectedBody)
        tree.focus_set()
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
        return 'break'

    def set_fa_text(x):
        G.log.replace('1.0', 'end', fa_to_string(G.search))
        G.log.focus_set()

    def do_search(x):
        fa = str_to_fa(G.log.get('1.0', 'end - 1c'))
        selranges = bw.tag_ranges('sel')
        if not selranges:
            st = en = len(bw.get('1.0', 'insert'))
        else:
            en = st = len(bw.get('1.0', 'sel.last'))
        #G.search = SearchState(fa, ltm.selectedPosition, en, en, G.search.in_head)
        s = G.search
        s.set_args(fa)
        s.curr_pos = ltm.selectedPosition
        s.pos_index = ltm.selectedIndex
        if fa.flags & FA_FLAGS['reverse']:
            m = s.find_next(s.sel_start, ltm.data)
        else:
            m = s.find_next(s.sel_end, ltm.data)
        if not m:
            print('not found')
        else:
            ltm.selectedPosition = s.curr_pos
            ltm.ensure_visible_index(s.pos_index)
            bw.setBodyTextSilently(ltm.selectedBody)
            bw.tag_remove('sel', '1.0', 'end')
            if s.in_head:
                print('found in head')
            else:
                print('found in body')
                bw.tag_add('sel', '1.0 + %dc'%m.start(), '1.0 + %dc'%m.end())
                bw.see('sel.first')
                bw.mark_set('insert', 'sel.first')
                bw.focus_set()
            draw_tree(tree, ltm)
            ltm.invalidate_visual()
    #@+node:vitalije.20180516131735.1: *3* next_node
    def next_node(x):
        gnx = ltm.select_next_node()
        if gnx:
            G.body.setBodyTextSilently(ltm.selectedBody)
        tree.focus_set()
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
        return 'break'

    #@+node:vitalije.20180516131739.1: *3* on_body_change
    def on_body_change(x):
        s = bw.get('1.0', 'end - 1c')
        bw.edit_modified(False)
        ltm.body_change(s)
        draw_tree(tree, ltm)

    #@+node:vitalije.20180516131743.1: *3* alt_left
    def alt_left(x):
        gnx = ltm.select_node_left()
        if gnx:
            G.body.setBodyTextSilently(ltm.selectedBody)
        tree.focus_set()
        draw_tree(tree, ltm)
        ltm.invalidate_visual()

    #@+node:vitalije.20180516131747.1: *3* alt_right
    def alt_right(x):
        gnx = ltm.select_node_right()
        if gnx:
            G.body.setBodyTextSilently(ltm.selectedBody)
        tree.focus_set()
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
    #@+node:vitalije.20180516131751.1: *3* mouse_wheel
    def mouse_wheel(ev):
        d = 1
        if os.name == 'nt':
            d = -1
        if ev.num == 4 or ev.delta < 0:
            topIndex.set(max(0, topIndex.get() - d))
        elif ev.num == 5 or ev.delta > 0:
            HR = 24; nr = rows_count(HR)
            ti = topIndex.get()
            cnt = len(tuple(ltm.display_items(ti, nr)))
            if cnt < nr - 1:
                return
            topIndex.set(topIndex.get() + d)

    #@+node:vitalije.20180516131800.1: *3* topIndex_write
    def topIndex_write(a, b, c):
        return draw_tree(tree, ltm)

    def show_sel():
        try:
            sel = ltm.visible_positions.index(ltm.selectedPosition) + 1
            nr = rows_count(24)
            i = topIndex.get()
            if nr + i < sel or i > sel - 1:
                topIndex.set(max(0, sel - nr // 2))
        except ValueError:
            pass
        G.app.after(30, show_sel)

    show_sel()
    #@+node:vitalije.20180518095655.1: *3* promote_sel
    def promote_sel(x):
        prep_undo()
        ltm.promote(ltm.selectedPosition)
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
    #@+node:vitalije.20180518095658.1: *3* pr_sel_ind
    def pr_sel_ind(x):
        print(ltm.selectedIndex)
        print(repr(bw.tag_ranges('sel')))
    #@+node:vitalije.20180518095702.1: *3* demote_sel
    def demote_sel(x):
        prep_undo()
        ltm.promote_children(ltm.selectedPosition)
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
    #@+node:vitalije.20180518095704.1: *3* move_right
    def move_right(x):
        prep_undo()
        ltm.indent_node(ltm.selectedPosition)
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
        return 'break'
    #@+node:vitalije.20180518095708.1: *3* move_left
    def move_left(x):
        prep_undo()
        ltm.dedent_node(ltm.selectedPosition)
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
        return 'break'
    #@+node:vitalije.20180518095711.1: *3* move_up
    def move_up(x):
        prep_undo()
        ltm.move_node_up(ltm.selectedPosition)
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
        return 'break'
    #@+node:vitalije.20180518095714.1: *3* move_down
    def move_down(x):
        prep_undo()
        ltm.move_node_down(ltm.selectedPosition)
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
        return 'break'
    #@+node:vitalije.20180614163543.1: *3* undo
    def undo(x):
        x = ltm.undo()
        if x:
            _restore_bodyw(x)
        ltm.invalidate_visual()
        draw_tree(tree, ltm)
        return 'break'

    def _restore_bodyw(x):
        ltm.selectedPosition = x[0]
        bw.setBodyTextSilently(ltm.selectedBody)
        bw.tag_remove('sel', '1.0', 'end')
        if x[1]:
            bw.tag_add('sel', x[1], x[2])
        bw.mark_set('insert', x[3])
        bw.see(x[3])
    #@+node:vitalije.20180614163712.1: *3* redo
    def redo(x):
        x = ltm.redo()
        if x:
            _restore_bodyw(x)
        bw.setBodyTextSilently(ltm.selectedBody)
        ltm.invalidate_visual()
        draw_tree(tree, ltm)
        return 'break'
    #@-others
    def expand_all(x):
        ltm.expand_all()
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
    def clonenode(x):
        ltm.clone_node(ltm.selectedPosition)
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
    topIndex.trace_add('write', topIndex_write)
    bw.bind(anglestr('TextModified'), on_body_change)
    bw.bind('<Alt-Key-Up>', prev_node)
    bw.bind('<Alt-Key-Down>', next_node)
    bw.bind('<Alt-Key-t>', lambda x:tree.focus_set())
    bw.bind('<Alt-Key-Left>', alt_left)
    bw.bind('<Alt-Key-Right>', alt_right)
    G.app.bind_all('<Alt-x>', pr_sel_ind) # for debugging purposes
    bw.bind_all('<Control-braceright>', promote_sel, add=True)
    bw.bind_all('<Control-braceleft>', demote_sel, add=True)
    bw.bind_all('<Shift-Control-Z>', redo, add=False)
    bw.bind_all('<Control-z>', undo, add=False)
    bw.bind_all('<F12>', expand_all)
    tree.bind('<Alt-Key-b>', lambda x:bw.focus_set())
    tree.bind('<Key-Return>', lambda x:bw.focus_set())
    tree.bind('<Key-Up>', prev_node)
    tree.bind('<Key-Down>', next_node)
    tree.bind('<Key-Left>', alt_left)
    tree.bind('<Shift-Left>', move_left)
    tree.bind('<Shift-Right>', move_right)
    tree.bind('<Shift-Up>', move_up)
    tree.bind('<Shift-Down>', move_down)
    tree.bind('<Key-Right>', alt_right)
    tree.bind('<MouseWheel>', mouse_wheel)
    tree.bind('<Button-4>', mouse_wheel)
    tree.bind('<Button-5>', mouse_wheel)
    G.app.bind_all('<Control-`>', clonenode)
    G.app.bind_all('<Shift-F10>', traverse_speed)
    G.app.bind_all('<F10>', speedtest)
    G.app.bind_all('<Control-f>', set_fa_text)
    G.app.bind_all('<F3>', do_search)
    topIndex.trace_add('write', topIndex_write)
#@+node:vitalije.20180614205557.1: ** prep_undo
def prep_undo():
    ltm = G.ltm
    bw = G.body
    x = [ltm.selectedPosition, None, None, bw.index('insert')]
    if bw.tag_ranges('sel'):
        x[1] = bw.index('sel.first')
        x[2] = bw.index('sel.last')
    ltm.pre_cmd(x)
#@+node:vitalije.20180515103819.1: ** bunch
class bunch:
    def __init__(self, **kw):
        self.__dict__.update(kw)

#@+node:vitalije.20180515103823.1: ** mk_icons
def mk_icons():
    G.icons = bunch(none=None)
    G.icons.plus = PhotoImage(**plusnode)
    G.icons.minus = PhotoImage(**minusnode)
    G.boxes = [PhotoImage(**boxKW(i)) for i in range(16)]

#@+node:vitalije.20180518115240.1: ** rows_count
def rows_count(h):
    return max(1, G.tree.winfo_height() // h)
#@+node:vitalije.20180518132651.1: ** click_pmicon
def click_pmicon(j):
    def switchExpand(x):
        ltm = G.ltm
        p = ltm.visible_positions[j + G.topIndex.get()]
        ltm.toggle(p)
        ltm.selectedPosition = p
        G.body.setBodyTextSilently(ltm.selectedBody)
        draw_tree(G.tree, ltm)
        ltm.invalidate_visual()
    return switchExpand

#@+node:vitalije.20180518132658.1: ** click_h
def click_h(j):
    def select_p(x):
        ltm= G.ltm
        p = ltm.visible_positions[j + G.topIndex.get()]
        ltm.selectedPosition = p
        G.body.setBodyTextSilently(ltm.selectedBody)
        draw_tree(G.tree, ltm)
        ltm.invalidate_visual()
    return select_p
#@+node:vitalije.20180515103828.1: ** draw_tree
def draw_tree(canv, ltm):
    HR = 24
    LW = 2 * HR
    count = rows_count(HR)
    items = list(canv.find('all'))
    #@+others
    #@+node:vitalije.20180518115341.1: *3* add selection highliter
    if len(items) < 1:
        # only first time we draw tree this is added
        items.append(canv.create_rectangle((0, -100, 300, -100+HR), 
            fill='#77cccc'))
    #@+node:vitalije.20180518115534.1: *3* drawing loop
    i = 1
    for j, dd in enumerate(ltm.display_items(G.topIndex.get(), count)):
        p, gnx, h, lev, pm, iconVal, sel, has_siblings = dd
        pmicon = getattr(G.icons, pm)
        i = j * 3 + 1
        x = lev * LW - 20
        y = j * HR + HR + 2
        if sel:
            canv.coords(items[0], 0, y - HR/2 - 2, canv.winfo_width(), y + HR/2 + 2)
            fg = '#000000'
        else:
            fg = '#a0a070'
        if i + 2 < len(items):
            # we are reconfiguring existing items
            if pmicon:
                canv.itemconfigure(items[i], image=pmicon)
                canv.coords(items[i], x, y)
            else:
                canv.coords(items[i], -200, y)
            canv.itemconfigure(items[i + 1], image=G.boxes[iconVal])
            canv.coords(items[i + 1], x + 20, y)
            canv.itemconfigure(items[i + 2], text=h, fill=fg)
            canv.coords(items[i + 2], x + 40, y)
        else:
            # we need to add more items to canvas
            items.append(canv.create_image(x, y, image=pmicon))
            items.append(canv.create_image(x + 20, y, image=G.boxes[iconVal]))
            items.append(canv.create_text(x + 40, y, text=h, anchor="w", fill=fg))
            canv.tag_bind(items[i], '<Button-1>', click_pmicon(j), add=False)
            canv.tag_bind(items[i + 1], '<Button-1>', click_h(j), add=False)
            canv.tag_bind(items[i + 2], '<Button-1>', click_h(j), add=False)
    #@-others

    # hide any extra item on canvas 
    for item in items[i + 3:]:
        canv.coords(item, 0, -200)


#@-others
if __name__ == '__main__':
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        print('usage: python miniTkLeo.py <leo document>')
        sys.exit()
    G = main(fname)
    connect_handlers()
    mk_icons()
    G.tree.after_idle(draw_tree, G.tree, G.ltm)
    G.tree.after_idle(G.tree.focus_set)
    G.tree.bind('<Expose>', lambda x: draw_tree(G.tree, G.ltm), True)
    G.app.mainloop()
#@-leo
