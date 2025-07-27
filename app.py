from flask import Flask, render_template, request, jsonify, session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Ganti dengan secret key yang aman

class TeoriKeputusan:
    def __init__(self):
        self.payoff_matrix = None
        self.alternatives = []
        self.states = []
        self.probabilities = []
        
    def set_payoff_matrix(self, alternatives, states, matrix):
        """Set matriks payoff dari web input"""
        self.alternatives = alternatives
        self.states = states
        self.payoff_matrix = np.array(matrix)
        
    def get_payoff_dataframe(self):
        """Mengembalikan matriks payoff dalam bentuk DataFrame"""
        if self.payoff_matrix is not None:
            return pd.DataFrame(self.payoff_matrix, 
                               index=self.alternatives, 
                               columns=self.states)
        return None
        
    def kondisi_ketidakpastian(self):
        """Pengambilan keputusan pada kondisi ketidakpastian"""
        results = {}
        
        # 1. MAXIMAX (Optimistik)
        max_values = np.max(self.payoff_matrix, axis=1)
        maximax_index = np.argmax(max_values)
        maximax_value = max_values[maximax_index]
        
        results['maximax'] = {
            'alternatif': self.alternatives[maximax_index],
            'nilai': float(maximax_value),
            'detail': {alt: float(val) for alt, val in zip(self.alternatives, max_values)}
        }
        
        # 2. MAXIMIN (Pesimistik)
        min_values = np.min(self.payoff_matrix, axis=1)
        maximin_index = np.argmax(min_values)
        maximin_value = min_values[maximin_index]
        
        results['maximin'] = {
            'alternatif': self.alternatives[maximin_index],
            'nilai': float(maximin_value),
            'detail': {alt: float(val) for alt, val in zip(self.alternatives, min_values)}
        }
        
        # 3. SAMA RATA (Equally Likely)
        avg_values = np.mean(self.payoff_matrix, axis=1)
        sama_rata_index = np.argmax(avg_values)
        sama_rata_value = avg_values[sama_rata_index]
        
        results['sama_rata'] = {
            'alternatif': self.alternatives[sama_rata_index],
            'nilai': float(sama_rata_value),
            'detail': {alt: float(val) for alt, val in zip(self.alternatives, avg_values)}
        }
        
        return results
        
    def kondisi_beresiko(self, probabilities):
        """Pengambilan keputusan pada kondisi beresiko"""
        self.probabilities = probabilities
        
        # Validasi probabilitas
        total_prob = sum(probabilities)
        if abs(total_prob - 1.0) > 0.001:
            return {'error': f'Total probabilitas = {total_prob:.3f} (harus = 1.0)'}
        
        # Hitung Expected Value (EV)
        expected_values = []
        calculations = []
        
        for i, alt in enumerate(self.alternatives):
            ev = 0
            calculation = []
            for j, state in enumerate(self.states):
                value = self.payoff_matrix[i][j] * probabilities[j]
                ev += value
                calculation.append({
                    'payoff': float(self.payoff_matrix[i][j]),
                    'prob': probabilities[j],
                    'result': float(value)
                })
            
            expected_values.append(ev)
            calculations.append(calculation)
        
        # Tentukan keputusan terbaik
        best_index = np.argmax(expected_values)
        best_alternative = self.alternatives[best_index]
        best_value = expected_values[best_index]
        
        return {
            'alternatif': best_alternative,
            'expected_value': float(best_value),
            'semua_ev': {alt: float(ev) for alt, ev in zip(self.alternatives, expected_values)},
            'calculations': dict(zip(self.alternatives, calculations))
        }
        
    def kondisi_pasti(self, state_index):
        """Pengambilan keputusan pada kondisi pasti"""
        if 0 <= state_index < len(self.states):
            chosen_state = self.states[state_index]
            payoffs = self.payoff_matrix[:, state_index]
            best_index = np.argmax(payoffs)
            best_alternative = self.alternatives[best_index]
            best_value = payoffs[best_index]
            
            return {
                'state': chosen_state,
                'alternatif': best_alternative,
                'nilai': float(best_value),
                'detail': {alt: float(val) for alt, val in zip(self.alternatives, payoffs)}
            }
        else:
            return {'error': 'Pilihan tidak valid!'}
            
    def generate_visualization(self, hasil_ketidakpastian=None, hasil_beresiko=None):
        """Generate visualisasi dan return sebagai base64 string"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Analisis Teori Keputusan', fontsize=16)
        
        # 1. Matriks Payoff
        im = axes[0, 0].imshow(self.payoff_matrix, cmap='RdYlGn', aspect='auto')
        axes[0, 0].set_title('Matriks Payoff')
        axes[0, 0].set_xlabel('States')
        axes[0, 0].set_ylabel('Alternatives')
        axes[0, 0].set_xticks(range(len(self.states)))
        axes[0, 0].set_yticks(range(len(self.alternatives)))
        axes[0, 0].set_xticklabels(self.states, rotation=45)
        axes[0, 0].set_yticklabels(self.alternatives)
        
        # Tambahkan nilai di dalam heatmap
        for i in range(len(self.alternatives)):
            for j in range(len(self.states)):
                axes[0, 0].text(j, i, f'{self.payoff_matrix[i, j]:.1f}',
                               ha='center', va='center', color='black', fontweight='bold')
        
        # 2. Perbandingan Metode Ketidakpastian
        if hasil_ketidakpastian:
            methods = ['Maximax', 'Maximin', 'Sama Rata']
            values = [hasil_ketidakpastian['maximax']['nilai'],
                     hasil_ketidakpastian['maximin']['nilai'],
                     hasil_ketidakpastian['sama_rata']['nilai']]
            
            bars = axes[0, 1].bar(methods, values, color=['green', 'red', 'blue'])
            axes[0, 1].set_title('Perbandingan Metode Ketidakpastian')
            axes[0, 1].set_ylabel('Nilai Payoff')
            
            # Tambahkan label nilai
            for bar, value in zip(bars, values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.2f}', ha='center', va='bottom')
        
        # 3. Expected Values (jika ada)
        if hasil_beresiko:
            alternatives = list(hasil_beresiko['semua_ev'].keys())
            ev_values = list(hasil_beresiko['semua_ev'].values())
            
            bars = axes[1, 0].bar(alternatives, ev_values, color='orange')
            axes[1, 0].set_title('Expected Values')
            axes[1, 0].set_ylabel('Expected Value')
            axes[1, 0].set_xticklabels(alternatives, rotation=45)
            
            # Tambahkan label nilai
            for bar, value in zip(bars, ev_values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.2f}', ha='center', va='bottom')
        
        # 4. Distribusi Payoff per Alternatif
        for i, alt in enumerate(self.alternatives):
            axes[1, 1].plot(self.states, self.payoff_matrix[i], 
                           marker='o', label=alt, linewidth=2)
        
        axes[1, 1].set_title('Distribusi Payoff per Alternatif')
        axes[1, 1].set_xlabel('States')
        axes[1, 1].set_ylabel('Payoff')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_string = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_string
        
    def analisis_sensitivitas(self):
        """Analisis sensitivitas untuk kondisi beresiko"""
        if not self.probabilities:
            return {'error': 'Belum ada data probabilitas untuk analisis sensitivitas'}
            
        # Variasi probabilitas state pertama
        prob_range = np.linspace(0.1, 0.9, 9)
        results = []
        
        for prob in prob_range:
            # Sesuaikan probabilitas state lainnya
            remaining_prob = 1 - prob
            temp_probs = [prob]
            
            # Distribusikan sisa probabilitas secara merata
            if len(self.states) > 1:
                for i in range(1, len(self.states)):
                    temp_probs.append(remaining_prob / (len(self.states) - 1))
            
            # Hitung EV untuk setiap alternatif
            expected_values = []
            for i in range(len(self.alternatives)):
                ev = sum(self.payoff_matrix[i][j] * temp_probs[j] 
                        for j in range(len(self.states)))
                expected_values.append(ev)
            
            best_index = np.argmax(expected_values)
            results.append({
                'prob': float(prob),
                'alternatif': self.alternatives[best_index],
                'ev': float(expected_values[best_index]),
                'all_ev': {alt: float(ev) for alt, ev in zip(self.alternatives, expected_values)}
            })
        
        return {
            'state_analyzed': self.states[0],
            'results': results
        }

# Global instance
decision = TeoriKeputusan()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input_matrix', methods=['POST'])
def input_matrix():
    try:
        data = request.json
        alternatives = data['alternatives']
        states = data['states']
        matrix = data['matrix']
        
        # Validasi input
        if len(alternatives) != len(matrix):
            return jsonify({'error': 'Jumlah alternatif tidak sesuai dengan matriks'})
        
        if len(states) != len(matrix[0]):
            return jsonify({'error': 'Jumlah state tidak sesuai dengan matriks'})
        
        # Set matriks payoff
        decision.set_payoff_matrix(alternatives, states, matrix)
        
        # Simpan dalam session
        session['alternatives'] = alternatives
        session['states'] = states
        session['matrix'] = matrix
        
        return jsonify({
            'success': True,
            'message': 'Matriks payoff berhasil disimpan',
            'dataframe': decision.get_payoff_dataframe().to_html(classes='table table-bordered')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/kondisi_ketidakpastian', methods=['GET'])
def kondisi_ketidakpastian():
    try:
        if decision.payoff_matrix is None:
            return jsonify({'error': 'Matriks payoff belum diinput'})
        
        results = decision.kondisi_ketidakpastian()
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/kondisi_beresiko', methods=['POST'])
def kondisi_beresiko():
    try:
        if decision.payoff_matrix is None:
            return jsonify({'error': 'Matriks payoff belum diinput'})
        
        data = request.json
        probabilities = data['probabilities']
        
        results = decision.kondisi_beresiko(probabilities)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/kondisi_pasti', methods=['POST'])
def kondisi_pasti():
    try:
        if decision.payoff_matrix is None:
            return jsonify({'error': 'Matriks payoff belum diinput'})
        
        data = request.json
        state_index = data['state_index']
        
        results = decision.kondisi_pasti(state_index)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/visualisasi', methods=['POST'])
def visualisasi():
    try:
        if decision.payoff_matrix is None:
            return jsonify({'error': 'Matriks payoff belum diinput'})
        
        data = request.json
        
        hasil_ketidakpastian = None
        hasil_beresiko = None
        
        if data.get('include_ketidakpastian'):
            hasil_ketidakpastian = decision.kondisi_ketidakpastian()
        
        if data.get('include_beresiko') and data.get('probabilities'):
            hasil_beresiko = decision.kondisi_beresiko(data['probabilities'])
        
        img_string = decision.generate_visualization(hasil_ketidakpastian, hasil_beresiko)
        
        return jsonify({
            'success': True,
            'image': img_string
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/analisis_sensitivitas', methods=['GET'])
def analisis_sensitivitas():
    try:
        if decision.payoff_matrix is None:
            return jsonify({'error': 'Matriks payoff belum diinput'})
        
        results = decision.analisis_sensitivitas()
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_matrix_info', methods=['GET'])
def get_matrix_info():
    try:
        if decision.payoff_matrix is None:
            return jsonify({'error': 'Matriks payoff belum diinput'})
        
        return jsonify({
            'alternatives': decision.alternatives,
            'states': decision.states,
            'matrix': decision.payoff_matrix.tolist(),
            'dataframe': decision.get_payoff_dataframe().to_html(classes='table table-bordered')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
