// HTML에서 호출할 함수: 입력 값을 사용하여 판매량을 예측하고 결과를 표시합니다.
async function predictSales() {
    // 모델 로드
    const model = await tf.loadLayersModel('model.json');

    // 입력 값을 가져옴
    const temperature = parseFloat(document.getElementById('temperature').value);
    const humidity = parseFloat(document.getElementById('humidity').value);

    // 예측을 위한 입력 데이터 형식으로 변환
    const inputTensor = tf.tensor2d([[temperature, humidity]]);

    // 판매량 예측
    const prediction = model.predict(inputTensor);

    // 예측 결과 표시
    const sales = prediction.dataSync()[0];
    document.getElementById('predictSales').textContent = sales.toFixed(2);

    // 모델 메모리 해제
    model.dispose();
}
