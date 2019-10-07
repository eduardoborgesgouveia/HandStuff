function [ numFrames,frameBoundaryTimes,numEvents ] = ...
    GetGeneralFrameInformation( AEDAT,timeStep)
%{
    Essa fun��o tem por objetivo retornar dados gerais sobre a forma��o de
    frames limitados por tempo.
    ---ENTRADA---
    AEDAT - arquivo AEDAT;
    timeStep - intervalo de tempo em que ser� acumulado os spikes para
    gera��o de um frame;
    ---SA�DA---
    numFrames - � a quantidade de frames dado o tempo total da grava��o e o
    timeStep;
    frameBoundaryTimes - vetor com limites iniciais e finais de tempo em
    que ser�o gerados frames da grava��o original
    numEvents - quantidade de eventos totais de uma grava��o
%}

[ t,to,tf,deltaT ] = GetTimeInformation( AEDAT );

numFrames = deltaT/timeStep;
frameTimes = to + timeStep*0.5 : timeStep : tf;
frameBoundaryTimes = [to frameTimes + timeStep * 0.5];
numEvents = length(t);

end

