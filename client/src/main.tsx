import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// 不使用 StrictMode：开发模式下它会重复执行挂载副作用，导致后端被重复拉起
ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
