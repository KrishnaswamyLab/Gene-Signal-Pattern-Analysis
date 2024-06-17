import unittest
import gspa
import numpy as np
import phate

class TestGSPA(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

        # Test input
        self.test_data = np.random.random(size=(1000,50))
        phate_op = phate.PHATE(verbose=False)
        phate_op.fit(self.test_data)

        self.bc_sample_idx = [1]*500 + [2]*500
        self.cell_type_assignments = ['cellA']*250 + ['cellB']*500 + ['cellC']*250
        self.cell_type = 'cellC'

        # Test setups
        self.gspa = gspa.GSPA(qr_decompose=False, pc_dim=20, embedding_dim=2,wavelet_J=3)
        self.gspa_qr = gspa.GSPA(pc_dim=20, embedding_dim=2)
        self.condensation = gspa.GSPA(perform_condensation=True, condensation_threshold=200, pc_dim=20, embedding_dim=2)
        self.batch_correction = gspa.GSPA(bc_sample_idx=self.bc_sample_idx, pc_dim=20, embedding_dim=2)
        self.input_graph = gspa.GSPA(graph=phate_op.graph, qr_decompose=False, pc_dim=20, embedding_dim=2, wavelet_J=3)
        self.input_diff_op = gspa.GSPA(diffusion_operator=phate_op.graph.diff_op, qr_decompose=False, pc_dim=20, embedding_dim=2, wavelet_J=3)
    
    def test_construct_graph(self):
        # Positive test case, no MS PHATE
        self.gspa.construct_graph(self.test_data)
        self.assertEqual(self.gspa.graph.N, self.test_data.shape[0])

    def test_construct_graph_MS_PHATE(self):
        # Positive test case, MS PHATE
        self.condensation.construct_graph(self.test_data)
        self.assertLessEqual(self.condensation.graph.N, 200)
   
    def test_construct_graph_BC(self):
        # Positive test case, BC
        self.batch_correction.construct_graph(self.test_data)
        self.assertEqual(self.batch_correction.graph.N, self.test_data.shape[0])

    def test_construct_diff_op_no_graph(self):
        with self.assertRaises(ValueError) as context:
            self.gspa.build_diffusion_operator()
        self.assertEqual(str(context.exception), "Graph not constructed. Run gspa_op.construct_graph(data) or initialize GSPA operator with graph")

    def test_construct_dict_no_graph(self):
        with self.assertRaises(ValueError) as context:
            self.gspa.build_wavelet_dictionary()
        self.assertEqual(str(context.exception), "Diffusion operator not constructed. Run gspa_op.build_diffusion_operator() or initialize GSPA operator with diffusion_operator")

    def test_dictionary(self):
        self.gspa.construct_graph(self.test_data)
        self.gspa.build_diffusion_operator()
        self.gspa.build_wavelet_dictionary()
        self.assertEqual(self.gspa.wavelet_dictionary.shape, (self.test_data.shape[0], self.test_data.shape[0] * self.gspa.wavelet_J))

    def test_dictionary_qr(self):
        self.gspa_qr.construct_graph(self.test_data)
        self.gspa_qr.build_diffusion_operator()
        self.gspa_qr.build_wavelet_dictionary()
        self.assertEqual(self.gspa_qr.wavelet_dictionary.shape[0], self.test_data.shape[0])
        self.assertLessEqual(self.gspa_qr.wavelet_dictionary.shape[1], self.test_data.shape[0] * self.gspa_qr.wavelet_J)
        
    def test_dictionary_MS_PHATE(self):
        self.condensation.construct_graph(self.test_data)
        self.condensation.build_diffusion_operator()
        self.condensation.build_wavelet_dictionary()
        self.assertLessEqual(self.condensation.wavelet_dictionary.shape[0], 200)

    def test_dictionary_input_graph(self):
        self.input_graph.build_diffusion_operator()
        self.input_graph.build_wavelet_dictionary()
        self.assertEqual(self.input_graph.wavelet_dictionary.shape, (self.test_data.shape[0], self.test_data.shape[0] * self.input_graph.wavelet_J))

    def test_dictionary_input_diff_op(self):
        self.input_diff_op.build_wavelet_dictionary()
        self.assertEqual(self.input_diff_op.wavelet_dictionary.shape, (self.test_data.shape[0], self.test_data.shape[0] * self.input_graph.wavelet_J))

    def test_get_gene_embeddings_no_wavelet(self):
        with self.assertRaises(ValueError) as context:
            self.gspa.get_gene_embeddings(self.test_data.T)
        self.assertEqual(str(context.exception), "Run gspa_op.build_wavelet_dictionary")

        with self.assertRaises(ValueError) as context:
            self.gspa.calculate_localization(self.test_data.T)
        self.assertEqual(str(context.exception), "Run gspa_op.build_wavelet_dictionary")

        with self.assertRaises(ValueError) as context:
            self.gspa.calculate_cell_type_specificity(cell_type_assignments=self.cell_type_assignments, cell_type=self.cell_type)
        self.assertEqual(str(context.exception), "Run gspa_op.build_wavelet_dictionary")

    def test_loc_no_signals(self):
        with self.assertRaises(ValueError) as context:
            self.gspa.construct_graph(self.test_data)
            self.gspa.build_diffusion_operator()
            self.gspa.build_wavelet_dictionary()
            self.gspa.calculate_localization()
        self.assertEqual(str(context.exception), "Provide signals to map to dictionary or run gspa_op.get_gene_embeddings")

        with self.assertRaises(ValueError) as context:
            self.gspa.construct_graph(self.test_data)
            self.gspa.build_diffusion_operator()
            self.gspa.build_wavelet_dictionary()
            self.gspa.calculate_cell_type_specificity(cell_type_assignments=self.cell_type_assignments, cell_type=self.cell_type)
        self.assertEqual(str(context.exception), "Provide signals to map to dictionary or run gspa_op.get_gene_embeddings")

    def test_get_gene_embeddings(self):
        self.gspa.construct_graph(self.test_data)
        self.gspa.build_diffusion_operator()
        self.gspa.build_wavelet_dictionary()
        out = self.gspa.get_gene_embeddings(self.test_data.T)
        self.assertEqual(out[0].shape, (self.test_data.shape[1], 2))
        self.assertEqual(out[1].shape, (self.test_data.shape[1], 20))

    def test_get_gene_embeddings(self):
        self.gspa.construct_graph(self.test_data)
        self.gspa.build_diffusion_operator()
        self.gspa.build_wavelet_dictionary()
        out = self.gspa.get_gene_embeddings(self.test_data.T)
        self.assertEqual(out[0].shape, (self.test_data.shape[1], 2))
        self.assertEqual(out[1].shape, (self.test_data.shape[1], 20))

    def test_get_gene_embeddings_MS_PHATE(self):
        self.condensation.construct_graph(self.test_data)
        self.condensation.build_diffusion_operator()
        self.condensation.build_wavelet_dictionary()
        out = self.condensation.get_gene_embeddings(self.test_data.T)
        self.assertEqual(out[0].shape, (self.test_data.shape[1], 2))
        self.assertEqual(out[1].shape, (self.test_data.shape[1], 20))

    def test_localization_with_gene_embeddings(self):
        self.gspa.construct_graph(self.test_data)
        self.gspa.build_diffusion_operator()
        self.gspa.build_wavelet_dictionary()
        self.gspa.get_gene_embeddings(self.test_data.T)
        out = self.gspa.calculate_localization()
        self.assertEqual(out.shape[0], self.test_data.shape[1])

    def test_localization_with_gene_embeddings_MS_PHATE(self):
        self.condensation.construct_graph(self.test_data)
        self.condensation.build_diffusion_operator()
        self.condensation.build_wavelet_dictionary()
        self.condensation.get_gene_embeddings(self.test_data.T)
        out = self.condensation.calculate_localization()
        self.assertEqual(out.shape[0], self.test_data.shape[1])

    def test_localization_without_gene_embeddings(self):
        self.gspa.construct_graph(self.test_data)
        self.gspa.build_diffusion_operator()
        self.gspa.build_wavelet_dictionary()
        out = self.gspa.calculate_localization(self.test_data.T[:20])
        self.assertEqual(out.shape[0], 20)

    def test_localization_without_gene_embeddings_MS_PHATE(self):
        self.condensation.construct_graph(self.test_data)
        self.condensation.build_diffusion_operator()
        self.condensation.build_wavelet_dictionary()
        self.condensation.get_gene_embeddings
        out = self.condensation.calculate_localization(self.test_data.T[:20])
        self.assertEqual(out.shape[0], 20)

    def test_cell_type_with_gene_embeddings(self):
        self.gspa.construct_graph(self.test_data)
        self.gspa.build_diffusion_operator()
        self.gspa.build_wavelet_dictionary()
        self.gspa.get_gene_embeddings(self.test_data.T)
        out = self.gspa.calculate_cell_type_specificity(cell_type_assignments=self.cell_type_assignments, cell_type=self.cell_type)
        self.assertEqual(out.shape[0], self.test_data.shape[1])

    def test_cell_type_with_gene_embeddings_MS_PHATE(self):
        self.condensation.construct_graph(self.test_data)
        self.condensation.build_diffusion_operator()
        self.condensation.build_wavelet_dictionary()
        self.condensation.get_gene_embeddings(self.test_data.T)
        out = self.condensation.calculate_cell_type_specificity(cell_type_assignments=self.cell_type_assignments, cell_type=self.cell_type)
        self.assertEqual(out.shape[0], self.test_data.shape[1])

    def test_cell_type_without_gene_embeddings(self):
        self.gspa.construct_graph(self.test_data)
        self.gspa.build_diffusion_operator()
        self.gspa.build_wavelet_dictionary()
        out = self.gspa.calculate_cell_type_specificity(cell_type_assignments=self.cell_type_assignments, cell_type=self.cell_type, signals=self.test_data.T[:20])
        self.assertEqual(out.shape[0], 20)

    def test_cell_type_without_gene_embeddings_MS_PHATE(self):
        self.condensation.construct_graph(self.test_data)
        self.condensation.build_diffusion_operator()
        self.condensation.build_wavelet_dictionary()
        self.condensation.get_gene_embeddings
        out = self.condensation.calculate_cell_type_specificity(cell_type_assignments=self.cell_type_assignments, cell_type=self.cell_type, signals=self.test_data.T[:20])
        self.assertEqual(out.shape[0], 20)
        


if __name__ == '__main__':
    unittest.main()
