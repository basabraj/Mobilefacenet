import os
import configparser

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

_camera_cfg = configparser.ConfigParser(interpolation=None)
_camera_cfg.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'camera.config'))
RTSP_URL = _camera_cfg.get('camera', 'rtsp_url')
LATENCY_MS = _camera_cfg.getint('camera', 'latency_ms')

Gst.init(None)

pipeline_in = Gst.Pipeline.new("pipeline")

source  = Gst.ElementFactory.make("rtspsrc",      "source")
depay   = Gst.ElementFactory.make("rtph265depay", "depay")
parser  = Gst.ElementFactory.make("h265parse",    "parser")
decoder = Gst.ElementFactory.make("avdec_h265",   "decoder")
convert = Gst.ElementFactory.make("videoconvert", "convert")
sink    = Gst.ElementFactory.make("autovideosink","sink")

source.set_property("location", RTSP_URL)
source.set_property("latency", LATENCY_MS)
sink.set_property("sync", False)

for element in [source, depay, parser, decoder, convert, sink]:
    pipeline_in.add(element)

def on_pad_added(src, pad):
    caps      = pad.get_current_caps() or pad.query_caps(None)
    structure = caps.get_structure(0)
    media     = structure.get_string("media")
    encoding  = structure.get_string("encoding-name")

    print(f"[PAD] media={media}, encoding={encoding}")

    if media != "video" or encoding != "H265":
        print("[PAD] Skipping non-H265-video pad")
        return

    sink_pad = depay.get_static_pad("sink")
    if not sink_pad.is_linked():
        ret = pad.link(sink_pad)
        print(f"[PAD] Linked: {ret}")

source.connect("pad-added", on_pad_added)

depay.link(parser)
parser.link(decoder)
decoder.link(convert)
convert.link(sink)

loop = GLib.MainLoop()

def on_bus_message(bus, message):
    t = message.type
    if t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[ERROR] {err}\n[DEBUG] {debug}")
        loop.quit()
    elif t == Gst.MessageType.EOS:
        print("[EOS] Stream ended")
        loop.quit()
    elif t == Gst.MessageType.STATE_CHANGED:
        if message.src == pipeline_in:
            old, new, _ = message.parse_state_changed()
            print(f"[STATE] {old.value_nick} -> {new.value_nick}")

bus = pipeline_in.get_bus()
bus.add_signal_watch()
bus.connect("message", on_bus_message)

pipeline_in.set_state(Gst.State.PLAYING)
print("Stream started. Press Ctrl+C to quit.")

try:
    loop.run()
except KeyboardInterrupt:
    pass
finally:
    pipeline_in.set_state(Gst.State.NULL)
