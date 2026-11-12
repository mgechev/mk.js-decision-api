/*
 * AI opponent for mk.js powered by the Google Decisions API
 * (https://aistudio.google.com/docs/decisions-api).
 *
 * Every "thinking" tick the current fight state is serialized to text and sent
 * to the `gemini-decision-1` model with a single `choice` question: "which
 * action should the AI fighter take next?". The returned probability
 * distribution over actions is then sampled (or arg-maxed on hard difficulty)
 * and turned into fighter moves.
 *
 * When no API key is configured, or while the network is failing, a local
 * rule-based brain is used so the game never stalls.
 */
;(function (mk) {
  'use strict';

  var DECISIONS_URL = 'https://generativelanguage.googleapis.com/v1beta/decisions';
  var MODEL = 'gemini-decision-1';
  var KEY_STORAGE = 'mkjs.geminiApiKey';

  mk.ai = {};

  mk.ai.getApiKey = function () {
    try { return localStorage.getItem(KEY_STORAGE) || ''; } catch (e) { return ''; }
  };

  mk.ai.setApiKey = function (key) {
    try {
      if (key) localStorage.setItem(KEY_STORAGE, key);
      else localStorage.removeItem(KEY_STORAGE);
    } catch (e) {}
  };

  /*
   * Actions the AI can choose from. Descriptions are what the model sees, so
   * they explain the in-game effect (damage, range, what beats what).
   */
  mk.ai.actions = {
    approach:     'Walk toward the opponent to get into striking range.',
    retreat:      'Walk away from the opponent to create space or escape pressure.',
    jump_in_kick: 'Jump toward the opponent and kick in the air (10 damage, long reach, good for closing distance).',
    jump_away:    'Jump backward away from the opponent to escape.',
    high_punch:   'Fast standing high punch (8 damage, short range). Misses a crouching opponent.',
    low_punch:    'Fast standing low punch (5 damage, short range). Hits a crouching opponent.',
    high_kick:    'Standing high kick (10 damage, short range). Misses a crouching opponent.',
    low_kick:     'Standing low kick (6 damage, short range). Hits a crouching opponent.',
    uppercut:     'Crouching uppercut (13 damage, slightly longer range, knocks the opponent down).',
    spin_kick:    'Slow roundhouse spin kick (13 damage, knocks the opponent down, leaves you exposed if it misses).',
    sweep:        'Crouching low kick (4 damage), safe poke while staying low.',
    crouch_punch: 'Crouching low punch (4 damage), safe poke while staying low.',
    block:        'Hold block; incoming damage is reduced to 20 percent.',
    crouch:       'Crouch to duck under high punches, high kicks and jump attacks.',
    wait:         'Stand still and wait for an opening.'
  };

  var DIFFICULTY = {
    easy:   { thinkEvery: 900, temperature: 1.6, reaction: 350 },
    normal: { thinkEvery: 450, temperature: 1.0, reaction: 120 },
    hard:   { thinkEvery: 200, temperature: 0,   reaction: 0 }
  };

  /* ---------------------------------------------------------------------- */
  /*                         Decisions API client                           */
  /* ---------------------------------------------------------------------- */

  mk.ai.DecisionsClient = function (options) {
    this.apiKey = options.apiKey;
    this.url = options.url || DECISIONS_URL;
    this.model = options.model || MODEL;
  };

  /**
   * @param {string|Array} input Shared context (string or content blocks).
   * @param {Object} questions Map of question key -> question definition.
   * @return {Promise<Object>} The `answers` map from the response.
   */
  mk.ai.DecisionsClient.prototype.decide = function (input, questions, signal) {
    return fetch(this.url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': this.apiKey
      },
      body: JSON.stringify({ model: this.model, input: input, questions: questions }),
      signal: signal
    }).then(function (res) {
      return res.json().then(function (body) {
        if (!res.ok) {
          var msg = (body && body.error && body.error.message) || res.statusText;
          throw new Error('Decisions API ' + res.status + ': ' + msg);
        }
        return body.answers || {};
      });
    });
  };

  /* ---------------------------------------------------------------------- */
  /*                              Helpers                                   */
  /* ---------------------------------------------------------------------- */

  function centerX(f) {
    return f.getX() + f.getVisibleWidth() / 2;
  }

  function distanceBetween(a, b) {
    return Math.abs(centerX(a) - centerX(b));
  }

  function rangeLabel(dist, reach) {
    if (dist <= reach) return 'close (inside striking range of punches and kicks)';
    if (dist <= reach * 1.5) return 'mid (just outside punch range, inside jump-attack range)';
    if (dist <= reach * 3) return 'far';
    return 'very far (opposite sides of the arena)';
  }

  /** Re-weights a probability map by temperature and samples from it. */
  function sample(probabilities, temperature) {
    var keys = Object.keys(probabilities);
    if (!keys.length) return null;
    if (!temperature) {
      return keys.reduce(function (best, k) {
        return probabilities[k] > probabilities[best] ? k : best;
      }, keys[0]);
    }
    var weights = keys.map(function (k) {
      return Math.pow(Math.max(probabilities[k], 1e-6), 1 / temperature);
    });
    var total = weights.reduce(function (a, b) { return a + b; }, 0);
    var r = Math.random() * total;
    for (var i = 0; i < keys.length; i += 1) {
      r -= weights[i];
      if (r <= 0) return keys[i];
    }
    return keys[keys.length - 1];
  }

  /* ---------------------------------------------------------------------- */
  /*                       Local fallback brain                             */
  /* ---------------------------------------------------------------------- */

  mk.ai.localPolicy = function (s) {
    var m = mk.moves.types, p = {};
    Object.keys(mk.ai.actions).forEach(function (k) { p[k] = 0.01; });
    var attacking = /kick|punch|uppercut/.test(s.opponentMove);
    if (s.distance > s.reach * 1.5) {
      p.approach = 0.6; p.jump_in_kick = 0.25; p.wait = 0.1;
    } else if (s.distance > s.reach) {
      p.approach = 0.5; p.jump_in_kick = 0.3; p.uppercut = 0.1;
    } else if (attacking) {
      p.block = 0.5; p.crouch = 0.2; p.uppercut = 0.2;
    } else if (s.opponentMove === m.SQUAT) {
      p.low_kick = 0.4; p.low_punch = 0.3; p.sweep = 0.2;
    } else {
      p.high_punch = 0.25; p.high_kick = 0.2; p.uppercut = 0.2;
      p.spin_kick = 0.1; p.low_kick = 0.1; p.block = 0.05;
    }
    if (s.selfLife < 25 && s.selfLife < s.opponentLife) p.retreat += 0.1;
    return p;
  };

  /* ---------------------------------------------------------------------- */
  /*                           AI controller                                */
  /* ---------------------------------------------------------------------- */

  /**
   * Player 1 (left) is human: keyboard and/or camera.
   * Player 2 (right) is driven by the Decisions API.
   *
   * Extra options:
   *  - ai.apiKey      Gemini API key (falls back to localStorage, then local brain)
   *  - ai.difficulty  'easy' | 'normal' | 'hard'
   *  - ai.onDecision  function (info) called after each decision (for HUD)
   */
  mk.controllers.AI = function (options) {
    mk.controllers.Basic.call(this, options);
    var ai = options.ai || {};
    this._aiOptions = ai;
    this._difficulty = DIFFICULTY[ai.difficulty] || DIFFICULTY.normal;
    var key = ai.apiKey || mk.ai.getApiKey();
    this._client = key ? new mk.ai.DecisionsClient({ apiKey: key, url: ai.url }) : null;
    this._history = [];
    this._timers = [];
    this._failures = 0;
  };

  mk.controllers.AI.prototype = new mk.controllers.Basic();

  mk.controllers.AI.prototype._initialize = function () {
    var self = this;
    this._player = 0;
    this._aiIndex = 1;
    this._addHandlers();
    this._trackOpponentHistory();
  };

  /** Round started (or single fight began): start thinking. */
  mk.controllers.AI.prototype._onFight = function () {
    var self = this;
    this._halt();
    this._running = true;
    this._failures = 0;
    this._history = [];
    this._loopTimer = setTimeout(function () { self._think(); }, 300);
  };

  /** Round over: stop thinking and cancel queued follow-up moves. */
  mk.controllers.AI.prototype._onHalt = function () {
    this._running = false;
    clearTimeout(this._loopTimer);
    this._timers.forEach(clearTimeout);
    this._timers = [];
    if (this._abort) this._abort.abort();
  };

  /** Lets non-keyboard inputs (camera) drive the human fighter. */
  mk.controllers.AI.prototype.controlPlayer = function (move) {
    var f = this.fighters[this._player], m = mk.moves.types;
    if (!f || !move) return;
    var cur = f.getMove() && f.getMove().type;
    if (move === m.STAND && cur === m.SQUAT) move = m.STAND_UP;
    f.setMove(move);
  };

  mk.controllers.AI.prototype.reset = function () {
    this._onHalt();
    clearInterval(this._historyTimer);
  };

  mk.controllers.AI.prototype._trackOpponentHistory = function () {
    var self = this, last = null;
    this._historyTimer = setInterval(function () {
      var f = self.fighters && self.fighters[self._player];
      if (!f || !f.getMove()) return;
      var t = f.getMove().type;
      if (t !== last) {
        self._history.push(t);
        if (self._history.length > 6) self._history.shift();
        last = t;
      }
    }, 60);
  };

  mk.controllers.AI.prototype._state = function () {
    var me = this.fighters[this._aiIndex],
      op = this.fighters[this._player],
      reach = op.getWidth() || 60;
    return {
      distance: Math.round(distanceBetween(me, op)),
      reach: reach,
      selfLife: Math.round(me.getLife()),
      opponentLife: Math.round(op.getLife()),
      selfMove: me.getMove().type,
      opponentMove: op.getMove().type,
      opponentJumping: op.isJumping(),
      selfNearWall: me.getX() < 40 || me.getX() > this.arena.width - me.getVisibleWidth() - 40,
      history: this._history.slice()
    };
  };

  mk.controllers.AI.prototype._describe = function (s) {
    return [
      'You are controlling the computer fighter in a 2D side-scrolling martial arts fighting game.',
      'Goal: reduce the opponent\'s health to 0 while protecting your own.',
      'Rules: attacks only land at close range (jump attacks and uppercut reach a bit further).',
      'A crouching fighter can only be hit by standing low punches and low kicks.',
      'Blocking reduces damage to 20%. Uppercut and spin kick knock the opponent down.',
      '',
      'Current state:',
      '- Distance between fighters: ' + s.distance + ' px, which is ' + rangeLabel(s.distance, s.reach) + '.',
      '- Your health: ' + s.selfLife + '/100. Opponent health: ' + s.opponentLife + '/100.',
      '- Opponent is currently doing: ' + s.opponentMove + (s.opponentJumping ? ' (airborne)' : '') + '.',
      '- You are currently doing: ' + s.selfMove + '.',
      '- Your back is ' + (s.selfNearWall ? 'against the wall' : 'not against a wall') + '.',
      '- Opponent\'s recent moves (oldest first): ' + (s.history.join(', ') || 'none') + '.',
      ''
    ].join('\n');
  };

  mk.controllers.AI.prototype._question = function () {
    return {
      next_action: {
        type: 'choice',
        instructions: 'Which single action gives the computer fighter the best chance of winning from this exact state? ' +
          'Prefer attacks only when they can actually connect at the current distance.',
        criteria: mk.ai.actions
      }
    };
  };

  mk.controllers.AI.prototype._think = function () {
    if (!this._running || !this.fighters) return;
    var self = this,
      me = this.fighters[this._aiIndex],
      op = this.fighters[this._player],
      d = this._difficulty;

    var next = function (delay) {
      if (!self._running) return;
      self._loopTimer = setTimeout(function () { self._think(); }, delay);
    };

    // Fight is over, or AI is mid-animation: try again shortly.
    if (me.getLife() <= 0 || op.getLife() <= 0) { this._running = false; return; }
    if (me._locked) { next(80); return; }

    var state = this._state(), started = Date.now();

    var act = function (probabilities, source) {
      var action = sample(probabilities, d.temperature);
      var info = {
        action: action,
        source: source,
        latency: Date.now() - started,
        probability: probabilities[action]
      };
      setTimeout(function () {
        self._perform(action);
        if (typeof self._aiOptions.onDecision === 'function') self._aiOptions.onDecision(info);
      }, d.reaction);
      next(Math.max(d.thinkEvery - info.latency, 60) + d.reaction);
    };

    if (!this._client || this._failures >= 3) {
      act(mk.ai.localPolicy(state), 'local');
      return;
    }

    this._abort = typeof AbortController === 'function' ? new AbortController() : null;
    this._client.decide(this._describe(state), this._question(), this._abort && this._abort.signal)
      .then(function (answers) {
        var a = answers.next_action;
        if (!a || !a.probabilities) throw new Error('No next_action in response');
        self._failures = 0;
        act(a.probabilities, 'decisions-api');
      })
      .catch(function (err) {
        if (!self._running) return;
        self._failures += 1;
        console.warn('[mk.ai] Decisions API failed, using local brain:', err.message);
        if (self._failures === 3) {
          // Retry the API after a cool-down instead of giving up forever.
          self._timers.push(setTimeout(function () { self._failures = 0; }, 15000));
        }
        act(mk.ai.localPolicy(state), 'local (API error)');
      });
  };

  /** Translates a high-level action into one or more fighter moves. */
  mk.controllers.AI.prototype._perform = function (action) {
    if (!this._running || !this.fighters) return;
    var self = this,
      m = mk.moves.types,
      me = this.fighters[this._aiIndex],
      op = this.fighters[this._player],
      towardRight = centerX(op) > centerX(me);

    var later = function (ms, fn) {
      self._timers.push(setTimeout(function () {
        if (self._running && self.fighters) fn();
      }, ms));
    };
    var hold = function (move, ms, release) {
      me.setMove(move);
      later(ms, function () {
        if (me.getMove().type === move) me.setMove(release || m.STAND);
      });
    };

    switch (action) {
      case 'approach':
        hold(towardRight ? m.WALK : m.WALK_BACKWARD, 350);
        break;
      case 'retreat':
        hold(towardRight ? m.WALK_BACKWARD : m.WALK, 350);
        break;
      case 'jump_in_kick':
        me.setMove(towardRight ? m.FORWARD_JUMP : m.BACKWARD_JUMP);
        later(330, function () {
          me.setMove(towardRight ? m.FORWARD_JUMP_KICK : m.BACKWARD_JUMP_KICK);
        });
        break;
      case 'jump_away':
        me.setMove(towardRight ? m.BACKWARD_JUMP : m.FORWARD_JUMP);
        break;
      case 'high_punch': me.setMove(m.HIGH_PUNCH); break;
      case 'low_punch':  me.setMove(m.LOW_PUNCH); break;
      case 'high_kick':  me.setMove(m.HIGH_KICK); break;
      case 'low_kick':   me.setMove(m.LOW_KICK); break;
      case 'uppercut':   me.setMove(m.UPPERCUT); break;
      case 'spin_kick':  me.setMove(m.SPIN_KICK); break;
      case 'sweep':
        me.setMove(m.SQUAT);
        later(140, function () { me.setMove(m.SQUAT_LOW_KICK); });
        later(500, function () { if (me.getMove().type === m.SQUAT) me.setMove(m.STAND_UP); });
        break;
      case 'crouch_punch':
        me.setMove(m.SQUAT);
        later(140, function () { me.setMove(m.SQUAT_LOW_PUNCH); });
        later(500, function () { if (me.getMove().type === m.SQUAT) me.setMove(m.STAND_UP); });
        break;
      case 'block':
        hold(m.BLOCK, 500);
        break;
      case 'crouch':
        hold(m.SQUAT, 500, m.STAND_UP);
        break;
      default:
        if (me.getMove().type !== m.STAND) me.setMove(m.STAND);
    }
  };

}(window.mk));
