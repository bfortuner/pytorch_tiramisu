import visdom
viz = visdom.Visdom()

def viz_plot_tst_trn(window, epoch, tst_val, trn_val, name='loss', env='main'):
    if window is None:
        return viz.line(
            X=np.array([epoch]),
            Y=np.array([[tst_val, trn_val]]),
            opts=dict(
                xlabel='epoch',
                ylabel=name,
                title=env+' '+name,
                legend=['Validation', 'Train']
            ),
            env=env
        )
    return viz.line(
        X=np.ones((1, 2)) * epoch,
        Y=np.expand_dims([tst_val, trn_val],0),
        win=window,
        update='append',
        env=env
    )

def viz_plot_img(window, tensor, env='main', title='Image'):
    '''
    This function draws an img on your Visdom web app.
    It takes as input an `CxHxW` tensor `img`
    The array values can be float in [0,1] or uint8 in [0, 255]'''
    np_img = decode_image(tensor)
    np_img = np.rollaxis(np_img, 2, 0)
    viz.image(
        np_img,
        opts=dict(title=title, caption='Silly image'),
        win=window,
        env=env
    )

def viz_plot_text(window, text, env='main'):
    if window is None:
        return viz.text(
            text,
            env=env
        )
    return viz.text(
        text,
        win=window,
        env=env
    )

def viz_plot_summary(window, epoch, tst_loss, trn_loss,
                       tst_err, trn_err, env='main'):
    txt = ("""Epoch: %d
        Train - Loss: %.3f Err: %.3f
        Test - Loss: %.3f Err: %.3f""" % (epoch,
        trn_loss, trn_err, tst_loss, tst_err))
    return viz_plot_text(window, txt, env)
