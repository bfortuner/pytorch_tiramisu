

class Experiment():
    def __init__(self, name, root):
        self.name = name
        self.root = os.path.join(root,name)
        self.epoch = 1
        self.best_val_loss = sys.maxsize
        self.best_val_loss_epoch = 1
        self.weights_dir = os.path.join(self.root, 'weights')
        self.history_dir = os.path.join(self.root, 'history')
        self.results_dir = os.path.join(self.root, 'results')
        self.latest_weights = os.path.join(self.weights_dir, 'latest_weights.pth')
        self.latest_optimizer = os.path.join(self.weights_dir, 'latest_optim.pth')
        self.best_weights_path = self.latest_weights
        self.best_optimizer_path = self.latest_optimizer
        self.train_history_fpath = os.path.join(self.history_dir, 'train.csv')
        self.val_history_fpath = os.path.join(self.history_dir, 'val.csv')
        self.test_history_fpath = os.path.join(self.history_dir, 'test.csv')
        self.loss_history = {
            'train': np.array([]),
            'val': np.array([]),
            'test': np.array([])
        }
        self.error_history = {
            'train': np.array([]),
            'val': np.array([]),
            'test': np.array([])
        }
        self.visdom_plots = self.init_visdom_plots()

    def init(self):
        print("Creating new experiment")
        self.init_dirs()
        self.init_history_files()

    def resume(self, model, optim, weights_fpath=None, optim_path=None):
        print("Resuming existing experiment")
        if weights_fpath is None:
            weights_fpath = self.latest_weights
        if optim_path is None:
            optim_path = self.latest_optimizer

        model, state = self.load_weights(model, weights_fpath)
        optim = self.load_optimizer(optim, optim_path)

        self.best_val_loss = state['best_val_loss']
        self.best_val_loss_epoch = state['best_val_loss_epoch']
        self.epoch = state['last_epoch']+1
        self.load_history_from_file('train')
        self.load_history_from_file('val')

        return model, optim

    def init_dirs(self):
        os.makedirs(self.weights_dir)
        os.makedirs(self.history_dir)
        os.makedirs(self.results_dir)

    def init_history_files(self):
        Path(self.train_history_fpath).touch()
        Path(self.val_history_fpath).touch()
        Path(self.test_history_fpath).touch()

    def init_visdom_plots(self):
        loss = self.init_viz_train_plot('loss')
        error = self.init_viz_train_plot('error')
        summary = self.init_viz_txt_plot('summary')
        return {
            'loss':loss,
            'error':error,
            'summary':summary
        }

    def init_viz_train_plot(self, title):
        return viz.line(
            X=np.array([1]),
            Y=np.array([[1, 1]]),
            opts=dict(
                xlabel='epoch',
                ylabel=title,
                title=self.name+' '+title,
                legend=['Train', 'Validation']
            ),
            env=self.name
        )

    def init_viz_txt_plot(self, title):
        return viz.text(
            "Initializing.. " + title,
            env=self.name
        )

    def viz_epochs(self):
        epochs = np.arange(1,self.epoch+1)
        return np.stack([epochs, epochs],1)

    def update_viz_loss_plot(self):
        loss = np.stack([self.loss_history['train'],
                         self.loss_history['val']],1)
        window = self.visdom_plots['loss']
        return viz.line(
            X=self.viz_epochs(),
            Y=loss,
            win=window,
            env=self.name,
            opts=dict(
                xlabel='epoch',
                ylabel='loss',
                title=self.name+' '+'loss',
                legend=['Train', 'Validation']
            ),
        )

    def update_viz_error_plot(self):
        error = np.stack([self.error_history['train'],
                         self.error_history['val']], 1)
        window = self.visdom_plots['error']
        return viz.line(
            X=self.viz_epochs(),
            Y=error,
            win=window,
            env=self.name,
            opts=dict(
                xlabel='epoch',
                ylabel='error',
                title=self.name+' '+'error',
                legend=['Train', 'Validation']
            )
        )

    def update_viz_summary_plot(self):
        trn_loss = self.loss_history['train'][-1]
        val_loss = self.loss_history['val'][-1]
        trn_err = self.error_history['train'][-1]
        val_err = self.error_history['val'][-1]
        txt = ("""Epoch: %d
            Train - Loss: %.3f Err: %.3f
            Test - Loss: %.3f Err: %.3f""" % (self.epoch,
            trn_loss, trn_err, tst_loss, tst_err))
        window = self.visdom_plots['summary']
        return viz.text(
            txt,
            win=window,
            env=self.name
        )

    def load_history_from_file(self, dset_type):
        fpath = os.path.join(self.history_dir, dset_type+'.csv')
        data = np.loadtxt(fpath, delimiter=',').reshape(-1, 3)
        self.loss_history[dset_type] = data[:,1]
        self.error_history[dset_type] = data[:,2]

    def append_history_to_file(self, dset_type, loss, error):
        fpath = os.path.join(self.history_dir, dset_type+'.csv')
        with open(fpath, 'a') as f:
            f.write('{},{},{}\n'.format(self.epoch, loss, error))

    def save_history(self, dset_type, loss, error):
        self.loss_history[dset_type] = np.append(
            self.loss_history[dset_type], loss)
        self.error_history[dset_type] = np.append(
            self.error_history[dset_type], error)
        self.append_history_to_file(dset_type, loss, error)

        if dset_type == 'val' and self.is_best_loss(loss):
            self.best_val_loss = loss
            self.best_val_loss_epoch = self.epoch

    def is_best_loss(self, loss):
        return loss < self.best_val_loss

    def save_weights(self, model, trn_loss, val_loss, trn_err, val_err):
        weights_fname = self.name+'-weights-%d-%.3f-%.3f-%.3f-%.3f.pth' % (
            epoch, trn_loss, trn_err, val_loss, val_err)
        weights_fpath = os.path.join(self.weights_dir, weights_fname)
        torch.save({
                'last_epoch': self.epoch,
                'trn_loss': trn_loss,
                'val_loss': val_loss,
                'trn_err': trn_err,
                'val_err': val_err,
                'best_val_loss': self.best_val_loss,
                'best_val_loss_epoch': self.best_val_loss_epoch,
                'experiment': self.name,
                'state_dict': model.state_dict()
            }, weights_fpath )
        shutil.copyfile(weights_fpath, self.latest_weights)
        if self.is_best_loss(val_loss):
            self.best_weights_path = weights_fpath

    def load_weights(self, model, fpath):
        print("loading weights '{}'".format(fpath))
        state = torch.load(fpath)
        model.load_state_dict(state['state_dict'])
        print (state['trn_err'], state['val_loss'], state['val_err'])
        print("loaded weights from experiment %s (last_epoch %d, trn_loss %s, trn_err %s, val_loss %s, val_err %s)" % (
                  self.name, state['last_epoch'], state['trn_loss'],
                    state['trn_err'], state['val_loss'], state['val_err']))
        return model, state

    def save_optimizer(self, optimizer, val_loss):
        optim_fname = self.name+'-optim-%d.pth' % (epoch)
        optim_fpath = os.path.join(self.weights_dir, optim_fname)
        torch.save({
                'last_epoch': self.epoch,
                'experiment': self.name,
                'state_dict': optimizer.state_dict()
            }, optim_fpath )
        shutil.copyfile(optim_fpath, self.latest_optimizer)
        if self.is_best_loss(val_loss):
            self.best_optimizer_path = optim_path

    def load_optimizer(self, optimizer, fpath):
        print("loading optimizer '{}'".format(fpath))
        optim = torch.load(fpath)
        optimizer.load_state_dict(optim['state_dict'])
        print("loaded optimizer from session {}, last_epoch {}"
              .format(optim['experiment'], optim['last_epoch']))
        return optim
