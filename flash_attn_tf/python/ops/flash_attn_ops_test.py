import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
from flash_attn_tf.python.ops import flash_attn_ops

class FlashAttnTest(test.TestCase):

  @test_util.run_gpu_only
  def testFlashAttnFwd(self):
    with self.test_session(use_gpu=True, force_gpu=True):
      with ops.device("/gpu:0"):
        batch_size = 2
        seqlen_q = 128
        seqlen_k = 128
        nheads = 4
        headdim = 32
        
        q = tf.random.normal((batch_size, seqlen_q, nheads, headdim), dtype=tf.float16)
        k = tf.random.normal((batch_size, seqlen_k, nheads, headdim), dtype=tf.float16)
        v = tf.random.normal((batch_size, seqlen_k, nheads, headdim), dtype=tf.float16)
        
        # Simple test to check if it runs without error
        # In a real test we would compare against a reference implementation
        out, lse = flash_attn_ops.flash_mha_fwd(
            q, k, v, 
            softmax_scale=1.0, 
            is_causal=False, 
            window_size_left=-1, 
            window_size_right=-1, 
            return_softmax=False,
            n=batch_size,
            l=seqlen_q,
            h=nheads,
            d=headdim,
            l_k=seqlen_k,
            h_k=nheads,
            dtype=0, # ignored, inferred from T
            seed=42
        )
        
        self.assertEqual(out.shape, (batch_size, seqlen_q, nheads, headdim))
        self.assertEqual(lse.shape, (batch_size, nheads, seqlen_q))

        # Force execution
        out_val = self.evaluate(out)
        self.assertEqual(out_val.shape, (batch_size, seqlen_q, nheads, headdim))


  @test_util.run_gpu_only
  def testFlashAttnBwd(self):
    with self.test_session(use_gpu=True, force_gpu=True):
      with ops.device("/gpu:0"):
        batch_size = 2
        seqlen_q = 128
        seqlen_k = 128
        nheads = 4
        headdim = 32
        
        q = tf.random.normal((batch_size, seqlen_q, nheads, headdim), dtype=tf.float16)
        k = tf.random.normal((batch_size, seqlen_k, nheads, headdim), dtype=tf.float16)
        v = tf.random.normal((batch_size, seqlen_k, nheads, headdim), dtype=tf.float16)
        dout = tf.random.normal((batch_size, seqlen_q, nheads, headdim), dtype=tf.float16)
        
        # Run forward first to get outputs needed for backward
        out, lse = flash_attn_ops.flash_mha_fwd(
            q, k, v, 
            softmax_scale=1.0, 
            is_causal=False, 
            window_size_left=-1, 
            window_size_right=-1, 
            return_softmax=False,
            n=batch_size,
            l=seqlen_q,
            h=nheads,
            d=headdim,
            l_k=seqlen_k,
            h_k=nheads,
            dtype=0,
            seed=42
        )

        dq, dk, dv = flash_attn_ops.flash_mha_bwd(
            dout, q, k, v, out, lse,
            p_dropout=0.0,
            softmax_scale=1.0,
            is_causal=False,
            window_size_left=-1,
            window_size_right=-1,
            deterministic=False,
            n=batch_size,
            l=seqlen_q,
            h=nheads,
            d=headdim,
            l_k=seqlen_k,
            h_k=nheads,
            seed=42
        )

        self.assertEqual(dq.shape, q.shape)
        self.assertEqual(dk.shape, k.shape)
        self.assertEqual(dv.shape, v.shape)

        # Force execution
        dq_val = self.evaluate(dq)
        self.assertEqual(dq_val.shape, (batch_size, seqlen_q, nheads, headdim))

if __name__ == '__main__':
  test.main()
