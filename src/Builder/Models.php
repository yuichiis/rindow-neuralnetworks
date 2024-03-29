<?php
namespace Rindow\NeuralNetworks\Builder;

use Rindow\NeuralNetworks\Model\Sequential;
use Rindow\NeuralNetworks\Model\ModelLoader;

class Models
{
    protected $builder;

    public function __construct(Builder $builder)
    {
        $this->builder = $builder;
    }

    public function Sequential(array $layers=null)
    {
        return new Sequential($this->builder,
                        $this->builder->utils()->HDA(),$layers);
    }

    public function loadModel($filepath)
    {
        $loader = new ModelLoader($this->builder,
                                        $this->builder->utils()->HDA());
        return $loader->loadModel($filepath);
    }

    public function modelFromConfig($config)
    {
        $loader = new ModelLoader($this->builder,
                                        $this->builder->utils()->HDA());
        return $loader->modelFromConfig($config);
    }
}
