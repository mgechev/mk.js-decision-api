/*
 * Camera move recognition for mk.js.
 *
 * Uses MediaPipe Pose Landmarker (runs fully in the browser, ~30fps) to track
 * 33 body landmarks, then classifies them into fighter moves:
 *
 *   step left / right      -> walk backward / forward
 *   squat                  -> squat
 *   jump                   -> jump (with a side step: forward/backward jump)
 *   punch (arm extended)   -> high punch (fist at head height) / low punch
 *   fist straight up       -> uppercut
 *   raise a knee / leg     -> low kick / high kick (leg above hip height)
 *   both fists at face     -> block
 *   squat + punch          -> squat low punch
 *
 * Usage:
 *   PoseInput.start({ container: el, onMove: function (move) {...} })
 */
;(function (w) {
  'use strict';

  var VERSION = '0.10.14';
  var CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@' + VERSION;
  var MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/' +
    'pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

  var L = {
    NOSE: 0, L_SHOULDER: 11, R_SHOULDER: 12, L_ELBOW: 13, R_ELBOW: 14,
    L_WRIST: 15, R_WRIST: 16, L_HIP: 23, R_HIP: 24, L_KNEE: 25, R_KNEE: 26,
    L_ANKLE: 27, R_ANKLE: 28
  };

  var EDGES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24],
    [23, 24], [23, 25], [25, 27], [24, 26], [26, 28]];

  // Tunables, expressed as fractions of the calibrated torso length.
  var T = {
    WALK: 0.45,        // horizontal hip displacement
    SQUAT: 0.30,       // hips dropping
    JUMP: 0.22,        // shoulders rising
    PUNCH_REACH: 0.75, // wrist horizontal distance from shoulder
    KICK_LOW: 0.35,    // ankle lifted above the other ankle
    KICK_HIGH: 0.75,
    BLOCK_X: 0.45,     // wrists close to the face
    MIN_VIS: 0.5,
    CALIBRATION_FRAMES: 25,
    ATTACK_COOLDOWN: 280
  };

  function mid(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }
  function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
  // MediaPipe extrapolates body parts that are outside the picture (e.g. feet
  // below the bottom edge). Those positions jitter, so they don't count.
  function inFrame(p) {
    return p.x >= -0.02 && p.x <= 1.02 && p.y >= -0.02 && p.y <= 1.02;
  }
  function visible(lm, ids) {
    return ids.every(function (i) {
      return lm[i] && inFrame(lm[i]) && (lm[i].visibility === undefined || lm[i].visibility > T.MIN_VIS);
    });
  }

  var PoseInput = {
    gestures: {
      STAND: 'stand', WALK_LEFT: 'walk-left', WALK_RIGHT: 'walk-right',
      SQUAT: 'squat', JUMP: 'jump', BLOCK: 'block',
      HIGH_PUNCH: 'high-punch', LOW_PUNCH: 'low-punch', UPPERCUT: 'uppercut',
      LOW_KICK: 'low-kick', HIGH_KICK: 'high-kick', SQUAT_PUNCH: 'squat-punch'
    }
  };

  /**
   * Pure classifier: landmarks + baseline -> { gesture, horizontal }.
   * Landmark x is in raw camera space; we mirror so "right" means the
   * player's screen-right (like looking in a mirror).
   */
  PoseInput.classify = function (lm, base) {
    var G = PoseInput.gestures,
      WALK = base.walk || T.WALK, SQUAT = base.squat || T.SQUAT,
      PUNCH = base.punch || T.PUNCH_REACH, KICK_LOW = base.kickLow || T.KICK_LOW,
      KICK_HIGH = base.kickHigh || T.KICK_HIGH;
    if (!visible(lm, [L.L_SHOULDER, L.R_SHOULDER, L.L_HIP, L.R_HIP])) return null;

    var shoulders = mid(lm[L.L_SHOULDER], lm[L.R_SHOULDER]),
      hips = mid(lm[L.L_HIP], lm[L.R_HIP]),
      t = base.torso,
      screenDx = base.hipX - hips.x,           // mirrored
      drop = hips.y - base.hipY,
      rise = base.shoulderY - shoulders.y,
      squatting = drop > SQUAT * t,
      jumping = !squatting && rise > T.JUMP * t && hips.y < base.hipY - T.JUMP * 0.6 * t,
      horizontal = screenDx > WALK * t ? 1 : (screenDx < -WALK * t ? -1 : 0);

    var nose = lm[L.NOSE];
    var arms = [[L.L_SHOULDER, L.L_ELBOW, L.L_WRIST], [L.R_SHOULDER, L.R_ELBOW, L.R_WRIST]];
    var wristsAtFace = 0, punch = null;

    arms.forEach(function (a) {
      if (!visible(lm, a)) return;
      var s = lm[a[0]], wr = lm[a[2]];
      var reachX = Math.abs(wr.x - s.x), aboveShoulder = wr.y < s.y + 0.1 * t;
      if (nose && aboveShoulder && Math.abs(wr.x - nose.x) < T.BLOCK_X * t && reachX < PUNCH * t) {
        wristsAtFace += 1;
      }
      if (nose && wr.y < nose.y - 0.35 * t && reachX < 0.5 * t) {
        punch = G.UPPERCUT;
      } else if (reachX > PUNCH * t && punch !== G.UPPERCUT) {
        var high = wr.y < s.y + 0.15 * t;
        if (punch !== G.HIGH_PUNCH) punch = high ? G.HIGH_PUNCH : G.LOW_PUNCH;
      }
    });

    var kick = null;
    if (visible(lm, [L.L_ANKLE, L.R_ANKLE]) && !jumping) {
      var la = lm[L.L_ANKLE], ra = lm[L.R_ANKLE];
      var lift = Math.abs(la.y - ra.y);
      var raised = la.y < ra.y ? la : ra;
      if (lift > KICK_HIGH * t || raised.y < hips.y + 0.25 * t) kick = G.HIGH_KICK;
      else if (lift > KICK_LOW * t) kick = G.LOW_KICK;
    } else if (visible(lm, [L.L_KNEE, L.R_KNEE]) && !jumping) {
      // Feet out of frame: fall back to knee lift.
      var kneeLift = Math.abs(lm[L.L_KNEE].y - lm[L.R_KNEE].y);
      if (kneeLift > Math.max(KICK_LOW, 0.3) * t) kick = G.LOW_KICK;
    }

    var gesture;
    if (wristsAtFace === 2) gesture = G.BLOCK;
    else if (punch === G.UPPERCUT) gesture = G.UPPERCUT;
    else if (squatting && punch) gesture = G.SQUAT_PUNCH;
    else if (kick) gesture = kick;
    else if (punch) gesture = punch;
    else if (jumping) gesture = G.JUMP;
    else if (squatting) gesture = G.SQUAT;
    else if (horizontal > 0) gesture = G.WALK_RIGHT;
    else if (horizontal < 0) gesture = G.WALK_LEFT;
    else gesture = G.STAND;

    return { gesture: gesture, horizontal: horizontal };
  };

  /**
   * Body measurements relative to the neutral baseline, in torso lengths.
   * Used by the guided calibration to learn the player's personal range.
   */
  PoseInput.metrics = function (lm, base) {
    if (!visible(lm, [L.L_SHOULDER, L.R_SHOULDER, L.L_HIP, L.R_HIP])) return null;
    var hips = mid(lm[L.L_HIP], lm[L.R_HIP]), t = base.torso, reach = 0, lift = 0;
    [[L.L_SHOULDER, L.L_WRIST], [L.R_SHOULDER, L.R_WRIST]].forEach(function (a) {
      if (visible(lm, a)) reach = Math.max(reach, Math.abs(lm[a[1]].x - lm[a[0]].x) / t);
    });
    if (visible(lm, [L.L_ANKLE, L.R_ANKLE])) lift = Math.abs(lm[L.L_ANKLE].y - lm[L.R_ANKLE].y) / t;
    else if (visible(lm, [L.L_KNEE, L.R_KNEE])) lift = Math.abs(lm[L.L_KNEE].y - lm[L.R_KNEE].y) / t;
    return {
      dx: (base.hipX - hips.x) / t,   // mirrored: + is the player's screen-right
      drop: (hips.y - base.hipY) / t,
      reach: reach,
      lift: lift
    };
  };

  /** Maps a gesture to an mk.js move type. */
  PoseInput.toMove = function (g, horizontal) {
    var m = w.mk.moves.types, G = PoseInput.gestures;
    switch (g) {
      case G.WALK_RIGHT: return m.WALK;
      case G.WALK_LEFT: return m.WALK_BACKWARD;
      case G.SQUAT: return m.SQUAT;
      case G.BLOCK: return m.BLOCK;
      case G.JUMP:
        return horizontal > 0 ? m.FORWARD_JUMP : (horizontal < 0 ? m.BACKWARD_JUMP : m.JUMP);
      case G.HIGH_PUNCH: return m.HIGH_PUNCH;
      case G.LOW_PUNCH: return m.LOW_PUNCH;
      case G.UPPERCUT: return m.UPPERCUT;
      case G.LOW_KICK: return m.LOW_KICK;
      case G.HIGH_KICK: return m.HIGH_KICK;
      case G.SQUAT_PUNCH: return m.SQUAT_LOW_PUNCH;
      default: return m.STAND;
    }
  };

  var ONE_SHOT = ['jump', 'high-punch', 'low-punch', 'uppercut', 'low-kick', 'high-kick', 'squat-punch'];

  var HOLD_FRAMES = 12, RELEASE_FRAMES = 8, STANCE_FRAMES = 30;

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function median(a) { var b = a.slice().sort(function (x, y) { return x - y; }); return b[Math.floor(b.length / 2)]; }

  /**
   * Guided calibration. Each step waits until the player holds the pose for
   * HOLD_FRAMES frames, records how far they actually moved, and derives a
   * personal threshold from it (a fraction of the player's own range), so
   * detection adapts to body size, camera distance and flexibility.
   */
  PoseInput.calibrationSteps = [
    { id: 'stance', title: 'Neutral stance',
      text: 'Stand in the middle of the view, whole body visible from head to feet, arms relaxed. Hold still.' },
    { id: 'left', title: 'Step left', text: 'Take a big step to your LEFT and hold.',
      metric: function (m) { return -m.dx; }, min: 0.25 },
    { id: 'right', title: 'Step right', text: 'Take a big step to your RIGHT and hold.',
      metric: function (m) { return m.dx; }, min: 0.25 },
    { id: 'squat', title: 'Squat', text: 'Squat down and hold.',
      metric: function (m) { return m.drop; }, min: 0.15 },
    { id: 'punch', title: 'Punch', text: 'Punch out to the side and hold your arm extended.',
      metric: function (m) { return m.reach; }, min: 0.5 },
    { id: 'kick', title: 'Kick', text: 'Raise one knee (or leg) as high as you would for a kick, and hold.',
      metric: function (m) { return m.lift; }, min: 0.2 },
    { id: 'block', title: 'Block', text: 'Put both fists in front of your face and hold.',
      check: function (lm, base) { var c = PoseInput.classify(lm, base); return c && c.gesture === 'block'; } }
  ];

  PoseInput._deriveThresholds = function (base, measured) {
    base.walk = clamp(0.55 * Math.min(measured.left, measured.right), 0.2, 0.8);
    base.squat = clamp(0.55 * measured.squat, 0.12, 0.5);
    base.punch = clamp(0.7 * measured.punch, 0.45, 1.1);
    base.kickLow = clamp(0.6 * measured.kick, 0.15, 0.6);
    base.kickHigh = Math.max(base.kickLow * 1.8, 0.6);
    return base;
  };

  /**
   * Starts the camera and pose model. Resolves once frames are being tracked.
   * Moves are only emitted after calibrate() has completed.
   */
  PoseInput.start = function (options) {
    var self = this;
    this._onMove = options.onMove;
    this._onStatus = options.onStatus || function () {};
    this._mode = 'idle';
    this._base = null;
    this._lastGesture = null;
    this._pending = null;
    this._pendingCount = 0;
    this._lastAttack = 0;
    this._buildDom(options.container);
    this._onStatus('Loading pose model…');

    return Promise.all([
      import(CDN + '/vision_bundle.mjs'),
      navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false })
    ]).then(function (r) {
      var vision = r[0], stream = r[1];
      self._stream = stream;
      self._video.srcObject = stream;
      return Promise.all([
        vision.FilesetResolver.forVisionTasks(CDN + '/wasm').then(function (fs) {
          return vision.PoseLandmarker.createFromOptions(fs, {
            baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' },
            runningMode: 'VIDEO',
            numPoses: 1
          });
        }),
        self._video.play()
      ]);
    }).then(function (r) {
      self._landmarker = r[0];
      self._onStatus('Camera ready.');
      self._loop();
    });
  };

  /**
   * Runs the guided calibration. `onUpdate(state)` is called every frame with
   * { index, step, progress (0..1), phase: 'hold' | 'release', hint }.
   * Resolves with the calibrated baseline + personal thresholds.
   */
  PoseInput.calibrate = function (onUpdate) {
    var self = this;
    this._mode = 'guided';
    this._base = null;
    return new Promise(function (resolve) {
      self._cal = {
        index: 0, phase: 'hold', frames: [], values: [], misses: 0,
        measured: {}, onUpdate: onUpdate || function () {}, resolve: resolve
      };
    });
  };

  /** Quick re-centre (neutral stance only), keeping personal thresholds. */
  PoseInput.recalibrate = function () {
    if (!this._base) return;
    this._keep = this._base;
    this._base = null;
    this._calib = [];
    this._mode = 'neutral';
    this._onStatus('Recalibrating: stand still in your neutral stance…');
  };

  PoseInput.stop = function () {
    cancelAnimationFrame(this._raf);
    if (this._stream) this._stream.getTracks().forEach(function (t) { t.stop(); });
    if (this._landmarker) this._landmarker.close();
  };

  /** Moves the camera view to another container (e.g. calibration -> game). */
  PoseInput.moveTo = function (container, className) {
    this._wrap.className = 'pose-wrap' + (className ? ' ' + className : '');
    container.appendChild(this._wrap);
  };

  PoseInput._buildDom = function (container) {
    var wrap = document.createElement('div');
    wrap.className = 'pose-wrap';
    var video = document.createElement('video');
    video.playsInline = true;
    video.muted = true;
    video.className = 'pose-video';
    var canvas = document.createElement('canvas');
    canvas.className = 'pose-canvas';
    canvas.width = 640;
    canvas.height = 480;
    var label = document.createElement('div');
    label.className = 'pose-label';
    wrap.appendChild(video);
    wrap.appendChild(canvas);
    wrap.appendChild(label);
    container.appendChild(wrap);
    this._wrap = wrap;
    this._video = video;
    this._canvas = canvas;
    this._ctx = canvas.getContext('2d');
    this._label = label;
  };

  PoseInput._loop = function () {
    var self = this, lastTime = -1;
    var frame = function () {
      self._raf = requestAnimationFrame(frame);
      var v = self._video;
      if (v.readyState < 2 || v.currentTime === lastTime) return;
      lastTime = v.currentTime;
      var res = self._landmarker.detectForVideo(v, performance.now());
      var lm = res.landmarks && res.landmarks[0];
      self._draw(lm);
      if (!lm) {
        self._label.textContent = 'no body detected';
        if (self._mode === 'guided') self._calUpdate(0, 'I can\'t see you. Step into the camera view.');
        return;
      }
      if (self._mode === 'guided') self._calibrateFrame(lm);
      else if (self._mode === 'neutral') self._neutralFrame(lm);
      else if (self._mode === 'play') self._process(lm);
    };
    frame();
  };

  PoseInput._calUpdate = function (progress, hint) {
    var c = this._cal;
    c.onUpdate({
      index: c.index,
      step: PoseInput.calibrationSteps[c.index],
      progress: progress,
      phase: c.phase,
      hint: hint || ''
    });
  };

  /** Neutral-stance capture shared by guided step 1 and quick recalibration. */
  PoseInput._stanceSample = function (lm, frames, needFeet) {
    if (!visible(lm, [L.L_SHOULDER, L.R_SHOULDER, L.L_HIP, L.R_HIP])) {
      frames.length = 0;
      return 'Move so your shoulders and hips are visible.';
    }
    if (needFeet && !visible(lm, [L.L_KNEE, L.R_KNEE])) {
      frames.length = 0;
      return 'I can\'t see your knees. Step further back from the camera.';
    }
    var s = mid(lm[L.L_SHOULDER], lm[L.R_SHOULDER]), h = mid(lm[L.L_HIP], lm[L.R_HIP]),
      sample = { hipX: h.x, hipY: h.y, shoulderY: s.y, torso: dist(s, h) };
    var first = frames[0];
    if (first && (Math.abs(first.hipX - sample.hipX) > 0.25 * first.torso ||
        Math.abs(first.hipY - sample.hipY) > 0.15 * first.torso)) {
      frames.length = 0;
      return 'Hold still…';
    }
    frames.push(sample);
    return '';
  };

  PoseInput._averageStance = function (frames) {
    var n = frames.length, avg = function (k) {
      return frames.reduce(function (a, c) { return a + c[k]; }, 0) / n;
    };
    return { hipX: avg('hipX'), hipY: avg('hipY'), shoulderY: avg('shoulderY'), torso: avg('torso') };
  };

  PoseInput._calibrateFrame = function (lm) {
    var c = this._cal, step = PoseInput.calibrationSteps[c.index];

    if (step.id === 'stance') {
      var hint = this._stanceSample(lm, c.frames, true);
      this._label.textContent = 'calibrating';
      if (c.frames.length >= STANCE_FRAMES) {
        this._base = this._averageStance(c.frames);
        return this._calNext();
      }
      return this._calUpdate(c.frames.length / STANCE_FRAMES, hint);
    }

    var m = PoseInput.metrics(lm, this._base);
    if (!m) return this._calUpdate(0, 'Move so your shoulders and hips are visible.');

    // Between steps: wait for the player to return to neutral.
    if (c.phase === 'release') {
      var why = Math.abs(m.dx) >= 0.2 ? 'Move back to where you stood at the start.' :
        m.drop >= 0.1 ? 'Stand up straight.' :
        m.reach >= 0.45 ? 'Lower your arms to your sides.' :
        m.lift >= 0.15 ? 'Put both feet on the floor.' : '';
      this._label.textContent = why ? 'back to neutral' : 'neutral';
      if (!why) {
        c.misses = 0;
        c.frames.push(1);
      } else if (++c.misses > 3) {   // a few jittery frames don't reset progress
        c.frames = [];
      }
      if (c.frames.length >= RELEASE_FRAMES) {
        c.phase = 'hold';
        c.frames = [];
        c.misses = 0;
        return this._calUpdate(0, '');
      }
      return this._calUpdate(c.frames.length / RELEASE_FRAMES, why || 'Good, hold still…');
    }

    var ok, value;
    if (step.check) {
      ok = step.check(lm, this._base);
    } else {
      value = step.metric(m);
      ok = value > step.min;
    }
    if (ok) {
      c.misses = 0;
      if (value !== undefined) c.values.push(value);
      c.frames.push(1);
    } else if (++c.misses > 2) {   // tolerate a couple of jittery frames
      c.frames = [];
      c.values = [];
    }
    if (c.frames.length >= HOLD_FRAMES) {
      if (c.values.length) c.measured[step.id] = median(c.values);
      if (step.id === 'kick') PoseInput._deriveThresholds(this._base, c.measured);
      return this._calNext();
    }
    this._label.textContent = step.title.toLowerCase();
    this._calUpdate(c.frames.length / HOLD_FRAMES, ok ? 'Good, hold it…' : '');
  };

  PoseInput._calNext = function () {
    var c = this._cal;
    c.index += 1;
    c.frames = [];
    c.values = [];
    c.misses = 0;
    c.phase = c.index === 1 ? 'hold' : 'release';
    if (c.index >= PoseInput.calibrationSteps.length) {
      this._mode = 'play';
      this._lastGesture = null;
      this._onStatus('Calibrated. Fight!');
      c.resolve(this._base);
      return;
    }
    this._calUpdate(0, '');
  };

  PoseInput._neutralFrame = function (lm) {
    var hint = this._stanceSample(lm, this._calib, false);
    this._label.textContent = 'recalibrating';
    if (hint) this._onStatus(hint);
    if (this._calib.length >= STANCE_FRAMES) {
      var b = this._averageStance(this._calib), k = this._keep;
      ['walk', 'squat', 'punch', 'kickLow', 'kickHigh'].forEach(function (p) { b[p] = k[p]; });
      this._base = b;
      this._mode = 'play';
      this._onStatus('Recalibrated.');
    }
  };

  PoseInput._process = function (lm) {
    var c = PoseInput.classify(lm, this._base);
    if (!c) return;
    this._label.textContent = this._lastGesture || 'stand';
    var g = c.gesture;

    // Require two consecutive frames for a gesture to filter jitter.
    if (g !== this._pending) {
      this._pending = g;
      this._pendingCount = 1;
      return;
    }
    this._pendingCount += 1;
    if (this._pendingCount !== 2) {
      // Held gestures (walk/squat/block) are re-sent occasionally in case
      // the fighter was locked in an animation when we first sent it.
      if (ONE_SHOT.indexOf(g) === -1 && this._pendingCount % 8 === 0) this._emit(g, c.horizontal);
      return;
    }
    if (g === this._lastGesture) return;
    if (ONE_SHOT.indexOf(g) !== -1) {
      var now = performance.now();
      if (now - this._lastAttack < T.ATTACK_COOLDOWN) return;
      this._lastAttack = now;
    }
    this._lastGesture = g;
    this._emit(g, c.horizontal);
  };

  PoseInput._emit = function (g, horizontal) {
    this._label.textContent = g;
    if (typeof this._onMove === 'function') this._onMove(PoseInput.toMove(g, horizontal), g);
  };

  PoseInput._draw = function (lm) {
    var ctx = this._ctx, cw = this._canvas.width, ch = this._canvas.height;
    ctx.save();
    ctx.translate(cw, 0);
    ctx.scale(-1, 1); // mirror
    ctx.drawImage(this._video, 0, 0, cw, ch);
    if (lm) {
      ctx.strokeStyle = '#3CD400';
      ctx.lineWidth = 3;
      EDGES.forEach(function (e) {
        var a = lm[e[0]], b = lm[e[1]];
        if (!a || !b) return;
        ctx.beginPath();
        ctx.moveTo(a.x * cw, a.y * ch);
        ctx.lineTo(b.x * cw, b.y * ch);
        ctx.stroke();
      });
      if (this._base) {
        ctx.strokeStyle = 'rgba(255,255,255,0.35)';
        ctx.setLineDash([6, 6]);
        var bx = this._base.hipX * cw, t = this._base.torso, wk = this._base.walk || T.WALK;
        [-wk, wk].forEach(function (k) {
          var x = bx + k * t * cw;
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ch); ctx.stroke();
        });
        ctx.setLineDash([]);
      }
    }
    ctx.restore();
  };

  w.PoseInput = PoseInput;

}(window));
