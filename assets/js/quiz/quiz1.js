let intentos = 0;
const maxIntentos = 3;
let timer;
let tiempoRestante = 60; // 60 segundos
let temporizadorIniciado = false;

// Estructura de las preguntas
const preguntas = [
  {
    id: 'q1',
    texto: '¿Qué protocolo se usa para la transferencia de archivos?',
    opciones: ['http', 'ftp', 'smtp'],
    respuestaCorrecta: 'ftp'
  },
  {
    id: 'q2',
    texto: '¿Cuál es la dirección de loopback en IPv4?',
    opciones: ['127.0.0.1', '192.168.1.1', '10.0.0.1'],
    respuestaCorrecta: '127.0.0.1'
  },
  {
    id: 'q3',
    texto: '¿Qué significa DHCP?',
    opciones: [
      'Dynamic Host Configuration Protocol',
      'Domain Host Control Protocol',
      'Data Hypertext Communication Protocol'
    ],
    respuestaCorrecta: 'Dynamic Host Configuration Protocol'
  }
];

// Generar preguntas dinámicamente
function generarPreguntas() {
  const form = document.getElementById('quiz-form');
  preguntas.forEach((pregunta, index) => {
    const preguntaElemento = document.createElement('p');
    preguntaElemento.innerHTML = `<strong>${index + 1}. ${
      pregunta.texto
    }</strong>`;
    form.appendChild(preguntaElemento);
    // Agregar salto de línea después del texto de la pregunta
    preguntaElemento.appendChild(document.createElement('br'));
    pregunta.opciones.forEach((opcion) => {
      const input = document.createElement('input');
      input.type = 'radio';
      input.name = pregunta.id;
      input.value = opcion;
      input.disabled = true; // Deshabilitar inicialmente
      preguntaElemento.appendChild(input);
      preguntaElemento.appendChild(document.createTextNode(opcion));
      preguntaElemento.appendChild(document.createElement('br'));
    });
  });

  // Crear botones
  const botonIniciar = document.createElement('button');
  botonIniciar.type = 'button';
  botonIniciar.id = 'iniciar';
  botonIniciar.textContent = 'Iniciar Intento';
  botonIniciar.onclick = iniciarIntento;
  form.appendChild(botonIniciar);

  const botonFinalizar = document.createElement('button');
  botonFinalizar.type = 'button';
  botonFinalizar.id = 'finalizar';
  botonFinalizar.textContent = 'Finalizar Intentos';
  botonFinalizar.style.display = 'none';
  botonFinalizar.onclick = finalizarIntentos;
  form.appendChild(botonFinalizar);

  const botonEnviar = document.createElement('button');
  botonEnviar.type = 'button';
  botonEnviar.id = 'enviar';
  botonEnviar.textContent = 'Enviar Respuestas';
  botonEnviar.style.display = 'none';
  botonEnviar.onclick = calcularPuntaje;
  form.appendChild(botonEnviar);
}

// Contador de intentos
function actualizarContadorIntentos() {
  document.getElementById('contador-intentos').textContent = intentos;
}

// Iniciar intento
function iniciarIntento() {
  if (intentos < maxIntentos) {
    intentos++;
    actualizarContadorIntentos();
    document.getElementById('iniciar').style.display = 'none'; // Ocultar botón de iniciar
    document.getElementById('finalizar').style.display = 'inline'; // Mostrar botón de finalizar
    document.getElementById('enviar').style.display = 'inline'; // Mostrar botón de enviar respuestas
    habilitarRespuestas(); // Habilitar las opciones de respuestas
    iniciarTemporizador(); // Iniciar el temporizador
  } else {
    alert('Has alcanzado el máximo de intentos.');
  }
}

// Habilitar las opciones de respuestas
function habilitarRespuestas() {
  let inputs = document.querySelectorAll('input[type="radio"]');
  inputs.forEach((input) => {
    input.disabled = false; // Habilitar las respuestas
  });
}

// Deshabilitar las opciones de respuestas
function deshabilitarRespuestas() {
  let inputs = document.querySelectorAll('input[type="radio"]');
  inputs.forEach((input) => {
    input.disabled = true; // Deshabilitar las respuestas
  });
}

// Temporizador
function iniciarTemporizador() {
  if (!temporizadorIniciado) {
    temporizadorIniciado = true;
    timer = setInterval(function () {
      if (tiempoRestante <= 0) {
        clearInterval(timer);
        calcularPuntaje();
      } else {
        let minutos = Math.floor(tiempoRestante / 60);
        let segundos = tiempoRestante % 60;
        document.getElementById('timer').textContent = `${minutos}:${
          segundos < 10 ? '0' + segundos : segundos
        }`;
        tiempoRestante--;
      }
    }, 1000);
  }
}

function calcularPuntaje() {
  // Detener temporizador si se llegó a 0 o el usuario hace clic en "Enviar Respuestas"
  clearInterval(timer);

  let respuestas = {};
  preguntas.forEach((pregunta) => {
    respuestas[pregunta.id] = document.querySelector(
      `input[name="${pregunta.id}"]:checked`
    )?.value;
  });

  let puntaje = 0;
  let respuestasIncorrectas = [];

  // Evaluar respuestas
  preguntas.forEach((pregunta) => {
    if (respuestas[pregunta.id] === pregunta.respuestaCorrecta) {
      puntaje++;
    } else {
      respuestasIncorrectas.push(pregunta.id);
    }
  });

  let porcentaje = (puntaje / preguntas.length) * 100;
  let mensaje =
    puntaje === preguntas.length
      ? '🎉 ¡Excelente! Respondiste todas las preguntas correctamente.'
      : puntaje === 0
      ? '❌ Todas las respuestas son incorrectas. Intenta de nuevo.'
      : `👍 ¡Bien hecho! Respondiste ${puntaje} de ${preguntas.length}.`;

  document.getElementById(
    'resultado'
  ).innerText = `Puntaje: ${puntaje}/${preguntas.length} - ${mensaje} (${porcentaje}%)`;

  // Mostrar respuestas incorrectas
  if (
    intentos >= maxIntentos ||
    document.getElementById('finalizar').style.display === 'none'
  ) {
    let respuestasIncorrectasTexto = respuestasIncorrectas
      .map(
        (id) => `${id}: ${preguntas.find((p) => p.id === id).respuestaCorrecta}`
      )
      .join(', ');
    document.getElementById('respuestas').style.display = 'block';
    document.getElementById(
      'respuestas'
    ).innerText = `Respuestas Incorrectas: ${respuestasIncorrectasTexto}`;
  }

  document.getElementById('finalizar').style.display = 'none'; // Ocultar botón de finalizar
  document.getElementById('enviar').style.display = 'none'; // Ocultar botón de enviar respuestas
  document.getElementById('nuevo-intento').style.display = 'block'; // Mostrar botón de nuevo intento
}

function finalizarIntentos() {
  clearInterval(timer); // Detener temporizador
  calcularPuntaje(); // Finalizar el intento
}

function reiniciarFormulario() {
  // No reiniciar el contador de intentos
  document.getElementById('quiz-form').reset();
  document.getElementById('resultado').innerText = '';
  document.getElementById('respuestas').innerText = '';
  document.getElementById('respuestas').style.display = 'none';
  document.getElementById('nuevo-intento').style.display = 'none';
  document.getElementById('iniciar').style.display = 'inline'; // Mostrar botón de iniciar
  document.getElementById('finalizar').style.display = 'none'; // Ocultar botón de finalizar
  document.getElementById('enviar').style.display = 'none'; // Ocultar botón de enviar respuestas
  deshabilitarRespuestas(); // Deshabilitar las opciones de respuestas
  tiempoRestante = 60; // Resetear temporizador
  temporizadorIniciado = false; // Permitir iniciar el temporizador nuevamente
  document.getElementById('timer').textContent = '00:00'; // Resetear el temporizador
}

// Inicializar formulario al cargar la página
window.onload = function () {
  generarPreguntas();
};
