# mk.js

A demo of [mk.js](https://github.com/mgechev/mk.js), an HTML5 canvas fighting game, with a computer opponent driven by the [Google Decisions API](https://aistudio.google.com/docs/decisions-api) (`gemini-decision-1`).

On every "thinking" tick, `src/ai.js` describes the fight as text (distance, health, both fighters' moves) and asks the Decisions API to pick one of 15 actions (approach, jump-in kick, uppercut, block, ...). You can also control your fighter with your body through the camera (MediaPipe Pose, runs locally).

**[Live Demo](https://mgechev.github.io/mk.js-decision-api/)**

## Usage

    python3 -m http.server 8080

1. Open `http://localhost:8080` (or the [GitHub Pages demo](https://mgechev.github.io/mk.js-decision-api/)) and choose **Play against AI**.
2. Paste a Gemini API key from https://aistudio.google.com/apikey. It's stored only in your browser's localStorage. Without a key, the opponent falls back to an offline rule-based AI.
3. Pick a difficulty. To fight with your body instead of the keyboard, tick **Control my fighter with my camera**. The camera needs `localhost` or HTTPS.

## License

[MIT](LICENSE)
