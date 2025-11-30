import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import json
import os

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from datetime import datetime
from torch.utils.data import DataLoader

from models.rnn import AudioRNN, LazyAudioRNNDataset, collate_fn_rnn
from utils import load_fold_paths

class RNNHyperparameterTuner:
    def __init__(self, 
                 data_cache_dir,
                 base_config,
                 n_trials=50,
                 objective_metric='val_loss',
                 direction='minimize',
                 study_name=None,
                 storage=None):
        
        self.data_cache_dir = data_cache_dir
        self.base_config = base_config.copy()
        self.n_trials = n_trials
        self.objective_metric = objective_metric
        self.direction = direction
        self.study_name = study_name or f"rnn_tuning_{datetime.now().strftime('%m%d_%H%M%S')}"
        self.storage = storage
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.results_dir = f"../tuning_results/{self.study_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def define_search_space(self, trial):
        config = self.base_config.copy()
        
        config['units'] = trial.suggest_categorical('units', [128, 256, 512])
        config['dense_units'] = trial.suggest_categorical('dense_units', [32, 64, 128, 256])
        
        config['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
        config['dropout'] = trial.suggest_float('dropout_dense', 0.2, 0.6, step=0.1)
        config['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
        
        config['lr'] = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        config['betas'] = [
            trial.suggest_float('beta1', 0.85, 0.95, step=0.05),
            trial.suggest_float('beta2', 0.99, 0.999, step=0.005)
        ]
        
        config['scheduler_step_size'] = trial.suggest_int('scheduler_step_size', 3, 6)
        config['scheduler_gamma'] = trial.suggest_float('scheduler_gamma', 0.3, 0.7, step=0.1)
        
        return config
    
    def objective(self, trial):
        config = self.define_search_space(trial)
        
        train_paths, train_labels, test_paths, test_labels = load_fold_paths(self.data_cache_dir)
        
        val_fold = 2
        train_folds = [f for f in range(1, 11) if f != val_fold]
        
        X_train = sum((train_paths[f] for f in train_folds), [])
        y_train = sum((train_labels[f] for f in train_folds), [])
        X_val = test_paths[val_fold]
        y_val = test_labels[val_fold]
        
        train_dataset = LazyAudioRNNDataset(X_train, y_train)
        val_dataset = LazyAudioRNNDataset(X_val, y_val)
        
        batch_size = config.get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=0, collate_fn=collate_fn_rnn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=0, collate_fn=collate_fn_rnn)
        
        model = AudioRNN(num_classes=config['num_classes'], config=config)
        model = model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            betas=config['betas'],
            eps=config.get('eps', 1e-8),
            weight_decay=config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['scheduler_step_size'],
            gamma=config['scheduler_gamma']
        )
        
        best_metric = float('inf') if self.direction == 'minimize' else float('-inf')
        patience = 5
        patience_counter = 0
        max_epochs = 20
        
        for epoch in range(max_epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for audio, lengths, labels in train_loader:
                audio = audio.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(audio, lengths)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = outputs.argmax(dim=1)
                train_correct += (pred == labels).sum().item()
                train_total += labels.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for audio, lengths, labels in val_loader:
                    audio = audio.to(self.device)
                    lengths = lengths.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(audio, lengths)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    pred = outputs.argmax(dim=1)
                    val_correct += (pred == labels).sum().item()
                    val_total += labels.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            current_metric = avg_val_loss if self.objective_metric == 'val_loss' else val_acc
            trial.report(current_metric, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            improved = False
            if self.direction == 'minimize':
                if current_metric < best_metric:
                    best_metric = current_metric
                    improved = True
            else:
                if current_metric > best_metric:
                    best_metric = current_metric
                    improved = True
            
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
            
            scheduler.step()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        del model, optimizer, scheduler, train_loader, val_loader
        del train_dataset, val_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return best_metric
    
    def run_tuning(self):
        print("\n" + "="*60)
        print("STARTING HYPERPARAMETER TUNING")
        print("="*60)
        print(f"Objective: {self.direction} {self.objective_metric}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Results directory: {self.results_dir}")
        print("="*60 + "\n")
        
        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            storage=self.storage,
            load_if_exists=True
        )
        
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self._save_results(study)
        
        return study
    
    def _save_results(self, study):
        best_trial = study.best_trial
        
        print("\n" + "="*60)
        print("TUNING COMPLETED")
        print("="*60)
        print(f"Best {self.objective_metric}: {best_trial.value:.4f}")
        print(f"Best trial number: {best_trial.number}")
        print("\nBest hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        
        best_config = self.base_config.copy()
        best_config.update(best_trial.params)
        
        if 'beta1' in best_trial.params and 'beta2' in best_trial.params:
            best_config['betas'] = [best_trial.params['beta1'], best_trial.params['beta2']]
            best_config.pop('beta1', None)
            best_config.pop('beta2', None)
        
        if 'dropout_dense' in best_trial.params:
            best_config['dropout'] = best_trial.params['dropout_dense']
            best_config.pop('dropout_dense', None)
        
        with open(os.path.join(self.results_dir, 'best_config.json'), 'w') as f:
            json.dump(best_config, f, indent=2)
        
        trials_df = study.trials_dataframe()
        trials_df.to_csv(os.path.join(self.results_dir, 'all_trials.csv'), index=False)
        
        stats = {
            'study_name': self.study_name,
            'objective_metric': self.objective_metric,
            'direction': self.direction,
            'n_trials': len(study.trials),
            'best_value': best_trial.value,
            'best_trial_number': best_trial.number,
            'best_params': best_trial.params
        }
        
        with open(os.path.join(self.results_dir, 'tuning_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        try:
            import matplotlib.pyplot as plt
            
            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.savefig(os.path.join(self.results_dir, 'optimization_history.png'))
            plt.close()
            
            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            plt.savefig(os.path.join(self.results_dir, 'param_importances.png'))
            plt.close()
            
        except Exception as e:
            print(f"Could not generate plots: {e}")
        
        print(f"\nResults saved to: {self.results_dir}")
        print("="*60 + "\n")


def tune_rnn_hyperparameters(data_cache_dir, 
                             base_config_path='../config/rnn.json',
                             n_trials=50,
                             objective='val_loss'):
    
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    direction = 'minimize' if objective == 'val_loss' else 'maximize'
    
    tuner = RNNHyperparameterTuner(
        data_cache_dir=data_cache_dir,
        base_config=base_config,
        n_trials=n_trials,
        objective_metric=objective,
        direction=direction
    )
    
    study = tuner.run_tuning()
    
    return study
