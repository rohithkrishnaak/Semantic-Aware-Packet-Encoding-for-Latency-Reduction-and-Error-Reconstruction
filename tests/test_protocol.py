import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from saep.models import IntScorer  # noqa: E402
from saep.protocol import (SAEC, SAEP_V2, Control, Packet, decode_control, decode_packet,  # noqa: E402
                           encode_control, encode_packet, image_to_tiles, peek_tag, tiles_to_image)


class ProtocolTest(unittest.TestCase):
    def test_header_sizes(self):
        self.assertEqual(SAEP_V2.size, 24)
        self.assertEqual(SAEC.size, 13)

    def test_packet_roundtrip(self):
        tag = np.arange(-16, 16, dtype=np.int8)
        pkt = Packet(7, 3, 16, 1, tag, bytes(range(192)))
        wire = encode_packet(pkt)
        self.assertEqual(len(wire), 24 + 32 + 192)
        out = decode_packet(wire)
        self.assertEqual((out.frame_id, out.pkt_index, out.pkt_count, out.tag_scheme_id), (7, 3, 16, 1))
        np.testing.assert_array_equal(out.tag, tag)
        self.assertEqual(out.payload, pkt.payload)
        fid, idx, ptag = peek_tag(wire)
        self.assertEqual((fid, idx), (7, 3))
        np.testing.assert_array_equal(ptag, tag)

    def test_corruption_rejected(self):
        wire = bytearray(encode_packet(Packet(1, 0, 16, 1, np.zeros(32, np.int8), b"x" * 192)))
        wire[-1] ^= 0xFF
        with self.assertRaises(ValueError):
            decode_packet(bytes(wire))
        with self.assertRaises(ValueError):
            decode_packet(bytes(wire[:-5]))
        with self.assertRaises(ValueError):
            decode_packet(b"XXXX" + bytes(wire[4:]))

    def test_control_roundtrip(self):
        c = decode_control(encode_control(Control(5, -1234, np.array([1, -2, 3], np.int8))))
        self.assertEqual((c.budget_per_frame, c.threshold), (5, -1234))
        np.testing.assert_array_equal(c.query, [1, -2, 3])
        self.assertIsNone(decode_control(encode_control(Control(16, 0))).query)

    def test_tiling_is_spatial_and_lossless(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, 3072, dtype=np.uint8)
        hwc = img.reshape(3, 32, 32).transpose(1, 2, 0)
        for grid in (2, 4, 8):
            tiles = image_to_tiles(img, grid)
            t = 32 // grid
            np.testing.assert_array_equal(tiles[grid + 1], hwc[t:2 * t, t:2 * t])
            back = tiles_to_image({i: tiles[i].tobytes() for i in range(grid * grid)}, grid)
            np.testing.assert_array_equal(back, hwc)

    def test_int_scorer_fit_ranks(self):
        rng = np.random.default_rng(0)
        tags = rng.integers(-127, 128, (2000, 16, 8)).astype(np.int8)
        true_w = rng.normal(size=8)
        util = tags.astype(float) @ true_w
        pos = np.broadcast_to(np.arange(16), (2000, 16))
        s = IntScorer.fit(tags, pos, util, 16, use_pos=False)
        self.assertGreater(np.corrcoef(s.score(tags, pos).ravel(), util.ravel())[0, 1], 0.99)
        self.assertEqual(s.w.dtype, np.int8)


if __name__ == "__main__":
    unittest.main()
