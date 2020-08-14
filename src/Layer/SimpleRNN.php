<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class SimpleRNN extends AbstractRNNLayer implements Layer
{
    use GenericUtils;
    protected $backend;
    protected $units;
    protected $activation;
    protected $useBias;
    protected $kernelInitializer;
    protected $biasInitializer;
    protected $returnSequence;
    protected $returnState;
    protected $statefull;

    protected $initalStates;
    protected $kernel;
    protected $recurrentKernel;
    protected $bias;
    protected $dKernel;
    protected $dBias;
    protected $inputs;

    public function __construct($backend,int $units, array $options=null)
    {
        extract($this->extractArgs([
            'input_shape'=>null,
            'activation'=>'tanh',
            'use_bias'=>true,
            'kernel_initializer'=>'sigmoid_normal',
            'recurrent_initializer'=>'sigmoid_normal',
            'bias_initializer'=>'zeros',
            'return_sequence'=>false,
            'return_state'=>false,
            'stateful'=>false,
            //'kernel_regularizer'=>null, 'bias_regularizer'=>null,
            //'activity_regularizer'=null,
            //'kernel_constraint'=null, 'bias_constraint'=null,
        ],$options));
        $this->backend = $K = $backend;
        $this->units = $units;
        $this->inputShape = $input_shape;
        if($use_bias) {
            $this->useBias = $use_bias);
        }
        $this->setActivation($activation);
        $this->kernelInitializerName = $kernel_initializer;
        $this->recurrentInitializerName = $recurrent_initializer;
        $this->biasInitializerName = $bias_initializer;
        $this->returnSequence=$return_sequence;
        $this->returnState = $return_state;
        $this->stateful = $statefull;
        $this->cell = new SimpleRNNCell(
            $this->backend,
            $this->units,
            ['activation'=>$this->activation,
            'use_bias'=>$this->useBias,
            'kernel_initializer'=>$this->kernelInitializerName,
            'recurrent_initializer'=>$this->recurrentInitializerName,
            'bias_initializer'=>$this->biasInitializerName,
            ]);
    }

    public function build(array $inputShape=null, array $options=null) : void
    {
        extract($this->extractArgs([
            'sampleWeights'=>null,
        ],$options));
        $K = $this->backend;
        $kernelInitializer = $this->kernelInitializer;
        $recurrentInitializer = $this->recurrentInitializer;
        $biasInitializer = $this->biasInitializer;

        $inputShape = $this->normalizeInputShape($inputShape);
        //if(count($inputShape)!=1) {
        //    throw new InvalidArgumentException(
        ///        'Unsuppored input shape: ['.implode(',',$inputShape).']');
        //}
        if(count($inputShape)!=2){
            throw new InvalidArgumentException('Unsuppored input shape.:['.implode(',',$inputShape).']');
        }
        $this->timesteps = $inputShape[0];
        $this->feature = $inputShape[1];
        $this->cell->build([$this->feature]);
        if($this->recurrentSequence){
            $this->outputShape = [$this->timesteps,$this->units];
        }else{
            $this->outputShape = [$this->units];
        }
    }

    public function getParams() : array
    {
        return $this->cell->getParams();
    }

    public function getGrads() : array
    {
        return $this->cell->getGrads();
    }

    public function getConfig() : array
    {
        return [
            'units' => $this->units,
            'options' => [
                'input_shape'=>$this->inputShape,
                'use_bias'=>$this->useBias;
                'activation'=>$this->activationName,
                'kernel_initializer' => $this->kernelInitializerName,
                'recurrent_initializer' => $this->recurrentInitializerName,
                'bias_initializer' => $this->biasInitializerName,
                'return_sequence'=>$this->returnSequence,
                'return_state'=>returnState,
                'stateful'=>$this->stateful,
            ]
        ];
    }

    protected function call(NDArray $inputs,bool $training, array $initalStates=null, array $options=null)
    {
        $K = $this->backend;
        [$batch,$timesteps,$feature]=$inputs->shape();
        if($initialStates===null&&
            $this->stateful) {
            $initialStates = $this->initialStates;
        }
        if($initialStates===null){
            $initialStates = [$K->zeros([$batch,$this->units])];
        }
        $outputs = null;
        if($this->returnSequence){
            $outputs=$K->zeros([$batch,$timesteps,$this->units]);
        }
        [$outputs,$states,$calcStates] = $K->rnn(
            [$this->cell,'forward'],
            $inputs,
            $initialStates,
            $outputs,
            null,
            $training,
        );
        $this->calcStates = $calcStates;
        if($this->stateful) {
            $this->initialStates = $states;
        }
        if($this->returnState){
            return [$outputs,$states]
        } else {
            return $outputs;
        }
    }

    protected function differentiate(NDArray $dOutputs, array $dNextStates=null) : array
    {
        $K = $this->backend;
        [$batch,$timesteps,$units]=$dOutputs->shape();
        $dInputs=$K->zeros([$batch,$timesteps,$this->feature]);
        if($dNextStates===null){
            $dNextStates = [$K->zeros([$batch,$this->units])];
        }

        $grads = $this->cell->getGrads();
        foreach($grads as $grad){
            $K->clear($grad);
        }
        [$dInputs,$dPrevStates] = $K->rnnBackward(
            [$this->cell,'backward'],
            $dOutputs,
            $dNextStates,
            $this->calcStates,
            $dInputs
        );
        $this->calcStates = null;
        if($this->returnState) {
            return [$dInputs, $dPrevStates];
        } else {
            return $dInputs;
        }
    }
}
