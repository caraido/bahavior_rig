function doneMessage = parVideoWorker(QueueFrom,QueueTo)
% the order of polls and sends is intrinsically tied to the order in
% parVideo2

%% Initialize
%get worker index and role from client, waiting until
hangTime = 1e4;
gpuWait = 1e-1;
doneMessage = false;
d = poll(QueueTo.Value,hangTime);
%d = poll(QueueFrom,hangTime);
thisIndex = d{1};
cameraIndex = d{2};
workerType = d{3};

%% Setup
bufferTime = 3; %time in seconds to wait between frames

switch workerType
    case 'acquisition'
        
        
        %delete(imaqfind);
        %imaqreset;
        % Configures acquisition to not stop if dropped frames occur
        %imaqmex('feature', '-gigeDisablePacketResend', true);
        
        % Detect cameras
        FLIRinfo = imaqhwinfo('mwspinnakerimaq');
        %FLIRinfo = imaqhwinfo('winvideo');
        
        %v = videoinput('winvideo', thisIndex,'YUY2_640x360');
        v = videoinput('mwspinnakerimaq', thisIndex); %resolution default
        s = v.Source;
        %set(s,'FrameRate','15.0000');
        %v.FrameGrabInterval = 15; %acquire at 2 fps
        
        % Logging properties
        %v.LoggingMode = 'disk'; %temporary
        v.LoggingMode = 'memory';
        
        %Trigger properties
        v.FramesPerTrigger = inf; %grabs continuously
        v.ReturnedColorSpace = 'grayscale';
        %triggerconfig(v,'manual');
        %v.FramesAcquiredFcnCount = bufferFrames; % when to send data to other workers
        %v.FramesAcquiredFcn = @sendWorkerFrames;
        %note that instead of framesacquiredfnc we can use a timerfcn...
        %we also may not need to send all of the data to the workers...
        
        %Camera properties
        % set through the source object, s
        % can also set an ROI to save?
    case 'processing'
        %connect to gpu?
        try
            gpuDevice(cameraIndex);
        catch ME
            send(QueueFrom,{thisIndex, 'Error initializing gpuDevice',ME});
            return
        end
end
%signal to client that we are done setting up
send(QueueFrom,{thisIndex, 'Ready'});

%% Acquisition Block
% await signal from client
while true
    pause(bufferTime);
    if strcmp(workerType,'acquisition') && strcmp(v.Logging,'on')
        sendWorkerFrames();
    end
    [msg,nohang] = poll(QueueTo.Value);
    %[msg,nohang] = poll(QueueFrom,hangTime);
    if nohang
        switch class(msg)
            case 'char'
                send(QueueFrom,{thisIndex,msg});
                switch msg
                    case 'beginAcq'
                        start(v);
                    case 'changeAcqType'
                        % stop(v) -> loggingMode() -> start(v)
                    case 'pauseAcq'
                        stop(v); %can restart with a new call to beginAcq
                    case 'endAcq'
                        stop(v);
                        delete(v);
                        delete(s);
                        delete(imaqfind);
                        imaqreset;
                        doneMessage = true;
                        return
                end
            case 'cell'
               switch msg{1}
                   case 'loadData'
                        %pause(gpuWait); %wait for data to come in...
%                         while ~nohang %keep waiting for data to come in... makes if statement useless
%                             [Data,nohang]=poll(QueueTo.Value);
%                             pause(gpuWait);
%                         end
%                        if nohang
                        try
                            send(QueueFrom,{thisIndex,msg{1}});
                            gData=gpuArray(squeeze(msg{2}));
                            bw = false(size(gData),'gpuArray');
                            for n=1:size(gData,3)
                                bw(:,:,n) = edge(gData(:,:,n));
                                %fprintf('.');
                            end
                            Data = gather(bw);
                            send(QueueFrom,{thisIndex,Data});
                        catch me %if you can
                            send(QueueFrom,{thisIndex,me});
                        end
%                        else
%                            send(QueueFrom,{thisIndex,'data missing'});
%                        end
               end
        end
        send(QueueFrom,{thisIndex,'Ready'});
    end
end


%% Helper functions
    function sendWorkerFrames()
        [frame,time] = getdata(v,v.FramesAvailable);
        % write time stamp to file
        send(QueueFrom,{thisIndex,frame,time});
    end

end