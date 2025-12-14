import { useState } from "react";
import { Client } from "@gradio/client";

function App() {
  const [text, setText] = useState("");
  const [model, setModel] = useState("Linear SVM");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    setResult(null);

    try {
      // Connect to your Hugging Face Space
      const client = await Client.connect(
        "jestoniandales25/ProfanityChecker"
      );

      // Call the Gradio function
      const response = await client.predict("/predict_profanity", {
        text: text,
        model_name: model
      });

      // Gradio returns results inside `.data`
      setResult(response.data);
    } catch (error) {
      console.error(error);
      setResult("Error connecting to API");
    }

    setLoading(false);
  };

  
  return (
    <div style={{ padding: 40, maxWidth: 600 }}>
      <h2>Profanity Detection</h2>

      <textarea
        rows={4}
        style={{ width: "100%" }}
        placeholder="Enter text"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <select
        value={model}
        onChange={(e) => setModel(e.target.value)}
        style={{ width: "100%", marginTop: 10 }}
      >
        <option>Linear SVM</option>
        <option>SGD (Hinge Loss)</option>
        <option>Passive Aggressive</option>
      </select>

      <button
        onClick={handleSubmit}
        disabled={loading}
        style={{ marginTop: 15 }}
      >
        {loading ? "Checking..." : "Check Text"}
      </button>

      {result && (
        <div style={{ marginTop: 20 }}>
          <strong>Result:</strong>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
