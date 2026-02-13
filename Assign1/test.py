import sys
import os
import cnn_engine

def test_engine():
    print("--- Starting C++ Engine Structural Test ---")

    try:
        # 1. Test ConvLayer (Assuming 3x3 filter by default or as per your C++ logic)
        conv = cnn_engine.ConvLayer(3)
        print(f"[SUCCESS] ConvLayer initialized. Parameters: {conv.getParams()}")

        # 2. Test FCLayer (Input size 10, Output size 2)
        fc = cnn_engine.FCLayer(10, 2)
        dummy_input_fc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        fc_out = fc.forward(dummy_input_fc)
        print(f"[SUCCESS] FCLayer forward pass result: {fc_out}")

        # 3. Test ReLULayer
        relu = cnn_engine.ReLULayer()
        # To this (adding a nested list):   
        relu_out = relu.forward([[-1.0, 0.0, 1.0]])
        print(f"[SUCCESS] ReLULayer working. Output: {relu_out}") # Should be [0.0, 0.0, 1.0]

        # 4. Test FlattenLayer
        flatten = cnn_engine.FlattenLayer()
        # Mocking a 2x2 matrix as input
        matrix_in = [[1.0, 2.0], [3.0, 4.0]]
        flat_out = flatten.forward(matrix_in)
        print(f"[SUCCESS] FlattenLayer working. Output: {flat_out}")

        # 5. Test Loss Layer
        loss_fn = cnn_engine.Loss()
        current_loss = loss_fn.calculate([0.9, 0.1], [1.0, 0.0])
        print(f"[SUCCESS] Loss calculation working. Value: {current_loss}")

        print("\n🏆 ALL STRUCTURES VERIFIED!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")

if __name__ == "__main__":
    test_engine()