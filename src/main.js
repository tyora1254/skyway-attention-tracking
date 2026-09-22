import { SkyWayContext, SkyWayRoom, SkyWayStreamFactory } from "@skyway-sdk/room";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const appId = "2f9fa80a-0064-4c68-b695-654120487301";
const secret = "BV6iX7LtVwMlGdM3+AUts8R7F7kfrx3vdHwPefJEVn4=";

// main.js 内に、カメラから映像を取得して、video タグにセットするコードを追加します。
// 1.即時実行の async 関数で全体を囲みます。これにより、非同期処理を await で記述できるようになります。以降の JavaScript の処理はこの即時実行関数内に記述します。
// 2.マイク音声とカメラ映像を取得し、それぞれを変数に分割代入します。
// 3.video 要素に映像（video）をセットします（audio は後ほど利用します）。
// 4.セットした映像を再生します。

(async () => {
  // 1
  const localVideo = document.getElementById("local-video");
  const overlayCanvas = document.getElementById("overlay-canvas");
  const ctx = overlayCanvas.getContext("2d");

  //HTML要素の取得
  const buttonArea = document.getElementById("button-area");
  const remoteMediaArea = document.getElementById("remote-media-area");
  const roomNameInput = document.getElementById("room-name");
  const myId = document.getElementById("my-id");
  const joinButton = document.getElementById("join");
  const leaveButton = document.getElementById('leave');

  // UI要素の取得
  const posBar = document.getElementById("posBar");
  const posText = document.getElementById("posText");
  const negBar = document.getElementById("negBar");
  const negText = document.getElementById("negText");
  const calibrateBtn = document.getElementById("calibrateBtn");

  // MediaPipe用の変数
  let faceLandmarker;
  let lastVideoTime = -1;
  let isMediaPipeReady = false;

  // キャリブレーションと推定用変数
  let calibratedPitch = null;
  let calibratedYaw = null;
  const YAW_RANGE = 20.0;
  const PITCH_RANGE = 15.0;
  let cursorXRatio = 0.5; // 初期値は画面中央(0.0~1.0)
  let cursorYRatio = 0.5;
  const smoothingFactor = 0.2;
  
  // 描画用の色変数
  let currentCursorColor = "rgba(255, 0, 0, 0.7)";
  let currentStrokeColor = "white";

  // 非表示用のビデオ要素(MediaPipe推論用)を作成
  const hiddenVideo = document.createElement('video');
  hiddenVideo.autoplay = true;
  hiddenVideo.playsInline = true;
  hiddenVideo.muted = true;

  // 1. MediaPipeの初期化関数
  async function initMediaPipe() {
      const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
              modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
              delegate: "GPU"
          },
          outputFacialTransformationMatrixes: true,
          outputFaceBlendshapes: true, // 表情データの出力をON
          runningMode: "VIDEO",
          numFaces: 1
      });
      isMediaPipeReady = true;
      console.log("MediaPipe is ready.");
  }

  // 行列からオイラー角の計算関数
  function getEulerAngles(matrix) {
      let sy = Math.sqrt(matrix[0] * matrix[0] + matrix[1] * matrix[1]);
      let singular = sy < 1e-6;
      let x, y, z;
      if (!singular) {
          x = Math.atan2(matrix[6], matrix[10]);
          y = Math.atan2(-matrix[2], sy);
          z = Math.atan2(matrix[1], matrix[0]);
      } else {
          x = Math.atan2(-matrix[9], matrix[5]);
          y = Math.atan2(-matrix[2], sy);
          z = 0;
      }
      return { pitch: x * (180 / Math.PI), yaw: y * (180 / Math.PI), roll: z * (180 / Math.PI) };
  }

  // キャリブレーション実行関数
  let currentRawPitch = null;
  let currentRawYaw = null;
  function doCalibrate() {
      if(currentRawPitch !== null && currentRawYaw !== null) {
          calibratedPitch = currentRawPitch;
          calibratedYaw = currentRawYaw;
          console.log(`Calibrated! Pitch: ${calibratedPitch.toFixed(1)}, Yaw: ${calibratedYaw.toFixed(1)}`);
      }
  }

  calibrateBtn.addEventListener('click', doCalibrate);
  window.addEventListener('keydown', (e) => { if (e.key === 'c' || e.key === 'C') doCalibrate(); });

  // メインの推論ループ関数
  async function predictWebcam() {
      if (!isMediaPipeReady) {
          window.requestAnimationFrame(predictWebcam);
          return;
      }

      let startTimeMs = performance.now();
      if (lastVideoTime !== hiddenVideo.currentTime) {
          lastVideoTime = hiddenVideo.currentTime;
          const results = faceLandmarker.detectForVideo(hiddenVideo, startTimeMs);

          let r = 255, g = 255, b = 255;
          let fillAlpha = 0;

          // --- 表情分析 (Blendshapes) ---
          if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
              const categories = results.faceBlendshapes[0].categories;
              const scores = {};
              categories.forEach(cat => scores[cat.categoryName] = cat.score);

              // 肯定スコア計算 (笑顔)
              const positiveScore = ((scores['mouthSmileLeft'] || 0) + (scores['mouthSmileRight'] || 0)) / 2.0;

              // 否定スコア計算 (しかめっ面・目を細める)
              const browDown = ((scores['browDownLeft'] || 0) + (scores['browDownRight'] || 0)) / 2.0;
              const squint = ((scores['eyeSquintLeft'] || 0) + (scores['eyeSquintRight'] || 0)) / 2.0;
              const negativeScore = (browDown + squint) / 2.0;

              // UIのゲージ更新
              if (posBar && posText) {
                  posBar.style.width = `${positiveScore * 100}%`;
                  posText.innerText = positiveScore.toFixed(2);
              }
              if (negBar && negText) {
                  negBar.style.width = `${negativeScore * 100}%`;
                  negText.innerText = negativeScore.toFixed(2);
              }

              // 閾値による色の変更
              if (positiveScore > 0.1 && positiveScore > negativeScore) {
                  r = 255; g = 50; b = 50;
                  fillAlpha = Math.min(1.0, positiveScore / 0.7) * 0.9; //0.7を最大として少し透明度下げる
              } else if (negativeScore > positiveScore && negativeScore > 0.05) {
                  r = 50; g = 150; b = 255;
                  fillAlpha = Math.min(1.0, negativeScore / 0.4) * 0.9; //0.4を最大として少し透明度下げる
              }
          }

          currentCursorColor = `rgba(${r}, ${g}, ${b}, ${fillAlpha})`;
          currentStrokeColor = `rgba(${r}, ${g}, ${b}, 0.8)`; // 枠線は常に少し表示しておく

          // --- 頭部方向の計算と座標更新 ---
          if (results.facialTransformationMatrixes && results.facialTransformationMatrixes.length > 0) {
              const matrix = results.facialTransformationMatrixes[0].data;
              const angles = getEulerAngles(matrix);
              
              currentRawPitch = angles.pitch;
              currentRawYaw = -angles.yaw;

              if (calibratedPitch !== null && calibratedYaw !== null) {
                  let deltaYaw = currentRawYaw - calibratedYaw;
                  let deltaPitch = currentRawPitch - calibratedPitch;

                  let normX = (deltaYaw + YAW_RANGE) / (YAW_RANGE * 2);
                  let normY = (deltaPitch + PITCH_RANGE) / (PITCH_RANGE * 2);
                  normX = Math.max(0, Math.min(1, normX));
                  normY = Math.max(0, Math.min(1, normY));

                  // スムージングを適用して座標割合を更新
                  cursorXRatio += (normX - cursorXRatio) * smoothingFactor;
                  cursorYRatio += (normY - cursorYRatio) * smoothingFactor;
                  
                  // SkyWay送信用変数へ統合
                  currentMousePos.x = cursorXRatio;
                  currentMousePos.y = cursorYRatio;
                  currentMousePos.color = currentCursorColor; // 色情報も送信データに含める
                  currentMousePos.stroke = currentStrokeColor;
              }
          }
      }
      
      // 自分が「聴衆(Audience)」なら自分の画面上にもローカルで円を描画する
      const roleElement = document.querySelector('input[name="role"]:checked');
      if (roleElement && roleElement.value === "audience") {
          drawCircle(currentMousePos.x, currentMousePos.y, currentMousePos.color, currentMousePos.stroke);
      }

      window.requestAnimationFrame(predictWebcam);
  }

  // 初期化の実行
  initMediaPipe();
  
  // --- 自分のマウス座標を常に記録しておく変数 ---
  let currentMousePos = { x: 0.5, y: 0.5 };
  // window.addEventListener("mousemove", (e) => {
  //   // 画面サイズが違ってもズレないように、画面に対する割合(0.0 ~ 1.0)で取得する
  //   currentMousePos.x = e.clientX / window.innerWidth;
  //   currentMousePos.y = e.clientY / window.innerHeight;
  // });

  // カメラを使う場合
  const { audio, video } = await SkyWayStreamFactory.createMicrophoneAudioAndCameraStream(); // 2

  // 取得したカメラ映像(video)を、MediaPipe推論用の非表示video要素にも流し込む
  const mediaStream = new MediaStream([video.track]);
  hiddenVideo.srcObject = mediaStream;
  hiddenVideo.addEventListener("loadeddata", predictWebcam);

  // 画面を使う場合
  const { audio: window_audio, video: window_video } = await SkyWayStreamFactory.createDisplayStreams(
    {
      audio: true, // 音声も取得する
      video: {
        displaySurface: 'monitor', // 画面全体の映像
      }
    }
  );

  // 選択式にするならそれ用の機構を作らないとこのif文意味ない、毎回videoで実行されてしまう
  // if (video) {
  //   video.attach(localVideo);
  //   await localVideo.play();
  // } else if (window_video) {
  //   window_video.attach(localVideo);
  //   await localVideo.play();
  // }

  joinButton.onclick = async () => {
    if (roomNameInput.value === "") return;

    // 選択された役割を取得
    const role = document.querySelector('input[name="role"]:checked').value;

    // 自分が「発表者」なら、自分の映像を大画面（local-video）にセット
    if (role === "presenter") {
      window_video.attach(localVideo);
      await localVideo.play();
    }

    //context: グローバル情報を管理するオブジェクト
    const context = await SkyWayContext.CreateForDevelopment(appId, secret);

    // 既に同じnameのroomが存在しなければ作成、存在すればそのroomを取得してroom変数へ代入
    const room = await SkyWayRoom.FindOrCreate(context, {
      name: roomNameInput.value,
    });

    //入室するとMemberオブジェクトが返ってくる
    const me = await room.join();

    //ID表示と metadata に役割(role)を持たせて入室
    myId.textContent = `${me.id} (${role === "presenter" ? "Presenter" : "Audience"})`;

    //音声と映像をpublish
    if(role === "presenter"){
      if (audio) await me.publish(audio, { type: "p2p" });
      if (video) await me.publish(video, { type: "p2p" });
      if (window_video) await me.publish(window_video, { type: "p2p" });

    }else{ //role === "Audience"は音声とカメラのみをpublish
      if (audio) await me.publish(audio, { type: "p2p" });
      if (video) await me.publish(video, { type: "p2p" });

      // データ通信用のストリームを作成してpublish
      const dataStream = await SkyWayStreamFactory.createDataStream();
      await me.publish(dataStream);

      // 高頻度で送りすぎてネットワークがパンクしないよう、50ms(1秒に20回)間隔で送信
      const sendIntervalId = setInterval(() => {
        // 座標データを文字列(JSON)に変換して送信
        dataStream.write(JSON.stringify(currentMousePos));
      }, 50);
    }    

    const subscribeAndAttach = (publication) => {
      // 3
      if (publication.publisher.id === me.id) return;

      const subscribeButton = document.createElement("button"); // 3-1
      subscribeButton.id = `subscribe-button-${publication.id}`;
      subscribeButton.textContent = `${publication.publisher.id}: ${publication.contentType}`;

      buttonArea.appendChild(subscribeButton);

      subscribeButton.onclick = async () => {
        // 3-2
        const { stream } = await me.subscribe(publication.id); // 3-2-1
        console.log(stream)

        // データ通信(座標)を受信した場合の処理 ---
        if (publication.contentType === "data") {
          stream.onData.add((data) => {
            const pos = JSON.parse(data);
            drawCircle(pos.x, pos.y, pos.color, pos.stroke);
          });
          return; // 映像ではないのでここで処理終了
        }

        let newMedia; // 3-2-2
        switch (publication.contentType) {
          case "video":
            // 相手が「発表者」で、自分が「聴衆」なら大画面にセット
            if (publication.publisher.metadata === "presenter" && role === "audience") {
              stream.attach(localVideo);
              await localVideo.play();
              return; // 大画面にセットしたので、小窓は作らずに終了
            }
            newMedia = document.createElement("video");
            newMedia.playsInline = true;
            newMedia.autoplay = true;
            break;
          case "audio":
            newMedia = document.createElement("audio");
            newMedia.controls = true;
            newMedia.autoplay = true;
            break;
          default:
            return;
        }
        newMedia.id = `media-${publication.id}`;
        stream.attach(newMedia); // 3-2-3
        remoteMediaArea.appendChild(newMedia);
      };
    };

    room.publications.forEach(subscribeAndAttach); // 1

    room.onStreamPublished.add((e) => {
      // 2
      subscribeAndAttach(e.publication);
    });

    //退室
    leaveButton.onclick = async () => {
      await me.leave();
      await room.dispose();

      myId.textContent = "";
      buttonArea.replaceChildren();
      remoteMediaArea.replaceChildren();
    };

    room.onStreamUnpublished.add((e) => {
      document.getElementById(`subscribe-button-${e.publication.id}`)?.remove();
      document.getElementById(`media-${e.publication.id}`)?.remove();
    });
  };

  function drawCircle(xRatio, yRatio, color = "rgba(255, 0, 0, 0.7)", stroke = "white") {
    // 描画のたびにCanvasの内部ピクセルサイズを、実際の表示サイズに合わせる
    overlayCanvas.width = overlayCanvas.clientWidth;
    overlayCanvas.height = overlayCanvas.clientHeight;
    
    // 一度キャンバスを綺麗に消去する
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    
    // 受信した割合座標を、現在のCanvasサイズに掛け合わせて実際のピクセル位置を計算
    const x = xRatio * overlayCanvas.width;
    const y = yRatio * overlayCanvas.height;
    
    // 円の描画
    ctx.beginPath();
    ctx.arc(x, y, 15, 0, Math.PI * 2); // 半径15pxの円
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = stroke;
    ctx.stroke();
  }
})(); // 1
