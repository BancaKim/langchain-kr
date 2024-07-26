

const StockChart = (() => {
    const init = () => {
        const ctx3 = document.getElementById('stockPriceChart3').getContext('2d');
        if (ctx3) {
            const stockDataElement = document.getElementById('stock-data');
            const stockDataRaw = JSON.parse(stockDataElement.textContent);

            const stockData = stockDataRaw.map(entry => ({
                x: new Date(entry.t),  // 날짜를 Date 객체로 변환
                o: entry.o,            // Open 가격
                h: entry.h,            // High 가격
                l: entry.l,            // Low 가격
                c: entry.c             // Close 가격
            }));

            new Chart(ctx3, {
                type: 'candlestick', // Candlestick 차트 유형
                data: {
                    datasets: [{
                        label: '주가',
                        data: stockData,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        color: 'rgba(75, 192, 192, 0.2)',
                        backgroundColor: function(context) {
                            const { dataIndex } = context;
                            const dataset = context.dataset.data[dataIndex];
                            return dataset.c > dataset.o ? 'rgba(75, 192, 192, 0.5)' : 'rgba(255, 99, 132, 0.5)';
                        },
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: '날짜'
                            },
                            type: 'time',
                            time: {
                                unit: 'day',
                                tooltipFormat: 'YYYY-MM-DD',
                                displayFormats: {
                                    day: 'YYYY-MM-DD'
                                }
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: '가격'
                            }
                        }
                    }
                }
            });
        } else {
            console.error('Canvas element not found.');
        }
    };

    return {
        init
    };
})();

// DOMContentLoaded 이벤트가 발생하면 초기화
document.addEventListener('DOMContentLoaded', () => {
    StockChart.init();
});